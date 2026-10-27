// C interface to Ramulator 2 for memory/ramulator.py (ctypes).
//
// One handle owns an External frontend and a memory system built from a
// fully expanded YAML config. Requests carry a caller-chosen tag; a tag is
// reported by atr_drain once the request completes (a read when its data
// returns, a write when its WR command issues -- Ramulator's convention).
// Errors never cross into Python as exceptions: calls that can fail return
// a status and leave the message in atr_last_error().

#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "ramulator/base/config.h"
#include "ramulator/base/factory.h"
#include "ramulator/base/request.h"
#include "ramulator/frontend/i_frontend.h"
#include "ramulator/memory_system/i_memory_system.h"

namespace {

std::string g_error;

struct Handle {
  std::unique_ptr<Ramulator::IFrontEnd> frontend;
  std::unique_ptr<Ramulator::IMemorySystem> memory;
  std::vector<int64_t> done;
  std::string stats;
};

}  // namespace

extern "C" {

__attribute__((visibility("default")))
const char* atr_last_error() { return g_error.c_str(); }

__attribute__((visibility("default")))
void* atr_create(const char* yaml) {
  try {
    auto cfg = Ramulator::Config::parse_config_string(yaml);
    auto* h = new Handle;
    h->frontend.reset(Ramulator::Factory::create_frontend(cfg));
    h->memory.reset(Ramulator::Factory::create_memory_system(cfg));
    h->frontend->connect_memory_system(h->memory.get());
    h->memory->connect_frontend(h->frontend.get());
    return h;
  } catch (const std::exception& e) {
    g_error = e.what();
    return nullptr;
  }
}

__attribute__((visibility("default")))
void atr_destroy(void* p) {
  auto* h = static_cast<Handle*>(p);
  if (h == nullptr) return;
  try {
    h->frontend->finalize();
    h->memory->finalize();
  } catch (...) {
  }
  delete h;
}

__attribute__((visibility("default")))
int atr_tx_bytes(void* p) { return static_cast<Handle*>(p)->memory->get_tx_bytes(); }

__attribute__((visibility("default")))
double atr_tck_ns(void* p) { return static_cast<Handle*>(p)->memory->get_tCK(); }

// 1: accepted, 0: the controller's queue is full (retry later), -1: error.
__attribute__((visibility("default")))
int atr_send(void* p, int is_write, uint64_t addr, int size_bytes, int64_t tag) {
  auto* h = static_cast<Handle*>(p);
  try {
    auto* done = &h->done;
    bool ok = h->frontend->receive_external_requests(
        is_write ? Ramulator::Request::Type::Write : Ramulator::Request::Type::Read,
        static_cast<Ramulator::Addr_t>(addr), 0,
        [done, tag](Ramulator::Request&) { done->push_back(tag); }, size_bytes);
    return ok ? 1 : 0;
  } catch (const std::exception& e) {
    g_error = e.what();
    return -1;
  }
}

// Tick the memory system n times (its own clock).
__attribute__((visibility("default")))
void atr_tick(void* p, int64_t n) {
  auto* h = static_cast<Handle*>(p);
  for (int64_t i = 0; i < n; i++) h->memory->tick();
}

// Copy up to `max` completed tags into `out`; returns how many.
__attribute__((visibility("default")))
int atr_drain(void* p, int64_t* out, int max) {
  auto* h = static_cast<Handle*>(p);
  int n = static_cast<int>(h->done.size());
  if (n > max) n = max;
  if (n > 0) {
    std::memcpy(out, h->done.data(), sizeof(int64_t) * n);
    h->done.erase(h->done.begin(), h->done.begin() + n);
  }
  return n;
}

// The memory system's statistics as YAML; valid until the next call.
__attribute__((visibility("default")))
const char* atr_stats(void* p) {
  auto* h = static_cast<Handle*>(p);
  try {
    h->memory->update_stats_recursive();
    std::ostringstream ss;
    h->memory->print_stats(ss);
    h->stats = ss.str();
  } catch (const std::exception& e) {
    h->stats = std::string("error: ") + e.what();
  }
  return h->stats.c_str();
}

}  // extern "C"
