#include "libfabric/libfabric_rail_manager.h"
#include "libfabric_mock_stubs.h"

#include <atomic>
#include <barrier>
#include <deque>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

#define TEST_REQUIRE(condition)                                              \
    do {                                                                     \
        if (!(condition)) {                                                  \
            throw std::runtime_error(std::string(#condition) + " at line " + \
                                     std::to_string(__LINE__));              \
        }                                                                    \
    } while (false)

extern "C" int
__wrap_numa_max_node() {
    return 0;
}

extern "C" int
__wrap_numa_num_configured_nodes() {
    return 1;
}

static int
closeNic(fid *handle) {
    auto *nic = reinterpret_cast<fid_nic *>(handle);
    free(nic->bus_attr);
    free(nic->link_attr);
    free(nic);
    return 0;
}

static fi_ops nic_ops = {.size = sizeof(fi_ops), .close = closeNic};

extern "C" int
__wrap_fi_getinfo(uint32_t, const char *, const char *, uint64_t, const fi_info *, fi_info **info) {
    *info = mock_fi_info_chain(1, 100ull * NIXL_LIBFABRIC_GIGA);
    (*info)->nic->fid.ops = &nic_ops;
    return 0;
}

extern "C" int
__wrap_fi_fabric(fi_fabric_attr *, fid_fabric **fabric, void *) {
    *fabric = mock_fabric_create();
    return 0;
}

struct Provider {
    size_t sends = 0;
    size_t cq_reads = 0;
    size_t minimum_cq_reads = 0;
    bool hold_completions = false;
    bool error_completion = false;
    std::deque<fi_cq_data_entry> completions;
    std::function<void()> on_send;
};

static Provider provider;

static ssize_t
sendData(fid_ep *,
         const void *,
         size_t length,
         void *,
         uint64_t data,
         fi_addr_t destination,
         void *context) {
    ++provider.sends;
    TEST_REQUIRE(provider.sends < 10000);
    if (provider.on_send) {
        provider.on_send();
    }
    if (destination == 1 || provider.cq_reads < provider.minimum_cq_reads) {
        return -FI_EAGAIN;
    }
    if (destination == 3) {
        return -FI_EIO;
    }
    provider.completions.push_back(
        {.op_context = context, .flags = FI_SEND, .len = length, .buf = nullptr, .data = data});
    return 0;
}

static ssize_t
readCompletions(fid_cq *, void *buffer, size_t capacity, fi_addr_t *sources) {
    ++provider.cq_reads;
    if (provider.error_completion && !provider.completions.empty()) {
        return -FI_EAVAIL;
    }
    if (provider.hold_completions || provider.completions.empty()) {
        return -FI_EAGAIN;
    }
    auto *entries = static_cast<fi_cq_data_entry *>(buffer);
    size_t count = 0;
    while (count < capacity && !provider.completions.empty()) {
        entries[count] = provider.completions.front();
        sources[count] = FI_ADDR_NOTAVAIL;
        provider.completions.pop_front();
        ++count;
    }
    return static_cast<ssize_t>(count);
}

static ssize_t
readError(fid_cq *, fi_cq_err_entry *entry, uint64_t) {
    TEST_REQUIRE(!provider.completions.empty());
    entry->op_context = provider.completions.front().op_context;
    entry->err = FI_EIO;
    provider.completions.pop_front();
    provider.error_completion = false;
    return 1;
}

struct Fixture {
    nixlLibfabricRailManager manager{0};
    nixlLibfabricRail &rail = manager.getRail(0);

    Fixture() {
        provider = {};
        TEST_REQUIRE(manager.getNumRails() == 1);
        TEST_REQUIRE(rail.initPostQueue(64));
        rail.setProgressThreadEnabled(true);
    }

    nixlLibfabricReq *
    request() {
        auto *request = rail.allocateControlRequest(8, LibfabricUtils::getNextXferId());
        TEST_REQUIRE(request != nullptr);
        return request;
    }

    nixl_status_t
    submit(nixlLibfabricReq *request,
           fi_addr_t destination,
           std::function<void(nixl_status_t)> callback = {}) {
        return manager.postControlMessage(nixlLibfabricRailManager::ControlMessageType::HANDSHAKE,
                                          request,
                                          destination,
                                          std::move(callback));
    }

    void
    progress() {
        auto status = rail.progressCompletionQueue();
        TEST_REQUIRE(status == NIXL_SUCCESS || status == NIXL_IN_PROG);
    }
};

static void
testHandshakeCallbackMakesProgress() {
    Fixture fixture;
    auto *response = fixture.request();
    auto *received = fixture.request();
    size_t callbacks = 0;
    fixture.rail.setHandshakeCallback([&](const std::string &) {
        TEST_REQUIRE(fixture.submit(response, 2, [&](nixl_status_t status) {
            TEST_REQUIRE(status == NIXL_SUCCESS);
            ++callbacks;
        }) == NIXL_SUCCESS);
        TEST_REQUIRE(provider.sends == 0);
    });
    provider.completions.push_back(
        {.op_context = &received->ctx,
         .flags = FI_RECV,
         .len = 8,
         .buf = received->buffer,
         .data = NIXL_MAKE_IMM_DATA(NIXL_LIBFABRIC_MSG_HANDSHAKE, 0, 0)});
    provider.minimum_cq_reads = 2;
    fixture.progress();
    TEST_REQUIRE(provider.sends == 1);
    TEST_REQUIRE(provider.cq_reads == 1);
    TEST_REQUIRE(response->in_use);
    TEST_REQUIRE(callbacks == 0);
    fixture.progress();
    TEST_REQUIRE(provider.sends == 2);
    TEST_REQUIRE(response->in_use);
    TEST_REQUIRE(callbacks == 0);
    fixture.progress();
    TEST_REQUIRE(!response->in_use);
    TEST_REQUIRE(callbacks == 1);
}

static void
testBoundedRetriesAndPeerFairness() {
    Fixture fixture;
    size_t completed = 0;
    for (size_t index = 0; index < NIXL_LIBFABRIC_CQ_BATCH_SIZE + 1; ++index) {
        TEST_REQUIRE(fixture.submit(fixture.request(), 1) == NIXL_SUCCESS);
    }
    auto *ready = fixture.request();
    TEST_REQUIRE(fixture.submit(ready, 2, [&](nixl_status_t status) {
        TEST_REQUIRE(status == NIXL_SUCCESS);
        ++completed;
    }) == NIXL_SUCCESS);
    bool added_new_requests = false;
    provider.on_send = [&] {
        if (!added_new_requests) {
            added_new_requests = true;
            for (size_t index = 0; index < NIXL_LIBFABRIC_CQ_BATCH_SIZE; ++index) {
                TEST_REQUIRE(fixture.submit(fixture.request(), 2) == NIXL_SUCCESS);
            }
        }
    };
    fixture.progress();
    TEST_REQUIRE(provider.sends == NIXL_LIBFABRIC_CQ_BATCH_SIZE);
    fixture.progress();
    TEST_REQUIRE(provider.sends == 2 * NIXL_LIBFABRIC_CQ_BATCH_SIZE);
    TEST_REQUIRE(provider.completions.size() == 1);
    fixture.progress();
    TEST_REQUIRE(completed == 1);
    TEST_REQUIRE(!ready->in_use);
    provider.on_send = {};
}

static void
testQueueBoundIncludesDetachedRequests() {
    Fixture fixture;
    for (size_t index = 0; index < NIXL_LIBFABRIC_RECV_POOL_SIZE; ++index) {
        TEST_REQUIRE(fixture.submit(fixture.request(), 1) == NIXL_SUCCESS);
    }
    auto *overflow = fixture.request();
    TEST_REQUIRE(fixture.submit(overflow, 1) == NIXL_ERR_BACKEND);
    TEST_REQUIRE(!overflow->in_use);
    provider.on_send = [&] {
        auto *reentrant = fixture.request();
        TEST_REQUIRE(fixture.submit(reentrant, 1) == NIXL_ERR_BACKEND);
        TEST_REQUIRE(!reentrant->in_use);
    };
    fixture.progress();
    TEST_REQUIRE(provider.sends == NIXL_LIBFABRIC_CQ_BATCH_SIZE);
    provider.on_send = {};
}

static void
testFailureAndCompletionOwnership() {
    for (bool completion_error : {false, true}) {
        Fixture fixture;
        auto *request = fixture.request();
        size_t callbacks = 0;
        TEST_REQUIRE(fixture.submit(request, completion_error ? 2 : 3, [&](nixl_status_t status) {
            TEST_REQUIRE(status == NIXL_ERR_BACKEND);
            TEST_REQUIRE(request->in_use);
            ++callbacks;
            TEST_REQUIRE(fixture.submit(fixture.request(), 1) == NIXL_SUCCESS);
        }) == NIXL_SUCCESS);
        fixture.progress();
        if (completion_error) {
            TEST_REQUIRE(request->in_use);
            TEST_REQUIRE(callbacks == 0);
            provider.error_completion = true;
            TEST_REQUIRE(fixture.rail.progressCompletionQueue() == NIXL_ERR_BACKEND);
        }
        TEST_REQUIRE(callbacks == 1);
        TEST_REQUIRE(!request->in_use);
        fixture.progress();
        TEST_REQUIRE(callbacks == 1);
    }
}

static void
testAcceptedRequestHeldUntilCompletion() {
    Fixture fixture;
    auto *request = fixture.request();
    TEST_REQUIRE(fixture.submit(request, 2) == NIXL_SUCCESS);
    provider.hold_completions = true;
    fixture.progress();
    for (size_t iteration = 0; iteration < 8; ++iteration) {
        fixture.progress();
        TEST_REQUIRE(request->in_use);
        TEST_REQUIRE(provider.sends == 1);
    }
    provider.hold_completions = false;
    fixture.progress();
    TEST_REQUIRE(!request->in_use);
}

static void
testCleanupReleasesOnlyUnpostedRequests() {
    Fixture fixture;
    auto *pending = fixture.request();
    auto *accepted = fixture.request();
    TEST_REQUIRE(fixture.submit(pending, 1) == NIXL_SUCCESS);
    TEST_REQUIRE(fixture.submit(accepted, 2) == NIXL_SUCCESS);
    fixture.progress();
    TEST_REQUIRE(pending->in_use);
    TEST_REQUIRE(accepted->in_use);
    fixture.rail.cleanup();
    TEST_REQUIRE(!pending->in_use);
    TEST_REQUIRE(accepted->in_use);
    fixture.rail.cleanup();
}

static void
testSynchronousFallbackAndOtherControlMessages() {
    Fixture fixture;
    fixture.rail.setProgressThreadEnabled(false);
    provider.minimum_cq_reads = 2;
    auto *request = fixture.request();
    TEST_REQUIRE(fixture.submit(request, 2) == NIXL_SUCCESS);
    TEST_REQUIRE(provider.cq_reads == 2);
    TEST_REQUIRE(provider.sends == 3);
    fixture.progress();
    TEST_REQUIRE(!request->in_use);
    fixture.rail.setProgressThreadEnabled(true);
    for (auto kind : {nixlLibfabricRailManager::ControlMessageType::NOTIFICATION,
                      nixlLibfabricRailManager::ControlMessageType::XFER_ERROR}) {
        auto *control = fixture.request();
        size_t before = provider.sends;
        TEST_REQUIRE(fixture.manager.postControlMessage(kind, control, 2) == NIXL_SUCCESS);
        TEST_REQUIRE(provider.sends == before + 1);
        fixture.progress();
        TEST_REQUIRE(!control->in_use);
    }
}

static void
testInvalidRequests() {
    Fixture fixture;
    TEST_REQUIRE(fixture.rail.queueHandshakeSend(0, 2, nullptr) == NIXL_ERR_INVALID_PARAM);
    for (size_t size : {size_t{0}, size_t{NIXL_LIBFABRIC_SEND_RECV_BUFFER_SIZE + 1}}) {
        auto *request = fixture.request();
        request->buffer_size = size;
        TEST_REQUIRE(fixture.submit(request, 2) == NIXL_ERR_INVALID_PARAM);
        TEST_REQUIRE(!request->in_use);
    }
    TEST_REQUIRE(provider.sends == 0);
}

static void
testConcurrentProducers() {
    Fixture fixture;
    constexpr size_t producer_count = 4;
    constexpr size_t per_producer = 64;
    std::barrier start(static_cast<std::ptrdiff_t>(producer_count + 1));
    std::atomic<size_t> submitted = 0;
    std::atomic<size_t> finished = 0;
    size_t completed = 0;
    std::vector<std::thread> producers;
    for (size_t producer = 0; producer < producer_count; ++producer) {
        producers.emplace_back([&] {
            start.arrive_and_wait();
            for (size_t index = 0; index < per_producer; ++index) {
                if (fixture.submit(fixture.request(), 2, [&](nixl_status_t status) {
                        TEST_REQUIRE(status == NIXL_SUCCESS);
                        ++completed;
                    }) == NIXL_SUCCESS) {
                    ++submitted;
                }
            }
            ++finished;
        });
    }
    start.arrive_and_wait();
    while (finished != producer_count) {
        fixture.progress();
        std::this_thread::yield();
    }
    for (auto &producer : producers) {
        producer.join();
    }
    for (size_t iteration = 0; iteration < producer_count * per_producer + 1; ++iteration) {
        fixture.progress();
    }
    TEST_REQUIRE(submitted == producer_count * per_producer);
    TEST_REQUIRE(completed == submitted);
    TEST_REQUIRE(provider.sends == submitted);
}

int
main() {
    fi_ep_msg_ops_stub.senddata = sendData;
    cq_ops_stub.readfrom = readCompletions;
    cq_ops_stub.readerr = readError;
    try {
        for (auto test : {testHandshakeCallbackMakesProgress,
                          testBoundedRetriesAndPeerFairness,
                          testQueueBoundIncludesDetachedRequests,
                          testFailureAndCompletionOwnership,
                          testAcceptedRequestHeldUntilCompletion,
                          testCleanupReleasesOnlyUnpostedRequests,
                          testSynchronousFallbackAndOtherControlMessages,
                          testInvalidRequests,
                          testConcurrentProducers}) {
            test();
        }
    }
    catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "All nine handshake progress tests passed\n";
    return 0;
}
