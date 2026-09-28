#pragma once
// the drafter's SWA KV cells on the host, as llama.cpp keeps them for the one sequence the server drafts on:
// llama_kv_cells (pos, used, seq_pos) and llama_kv_cache's seq_rm, find_slot (cont = false: the server's prepare),
// apply_ubatch and get_n_kv (src/llama-kv-cache.cpp). The engine writes the cells the server would, masks the same window
// and runs the same n_kv. `dirty` collects the cells whose position changed since the device table last took them.
#include <algorithm>
#include <cstdint>
#include <set>
#include <utility>
#include <vector>

struct Ring {
    uint32_t size = 0, n_swa = 0, head = 0;
    std::vector<int32_t> pos;                        // -1: empty
    std::set<uint32_t> used;                         // llama_kv_cells::used
    std::set<std::pair<int32_t, uint32_t>> sp;       // llama_kv_cells::seq_pos[0]: (position, cell)
    std::vector<uint32_t> dirty;
    std::vector<uint8_t>  is_dirty;

    // the server's table: cells [0, n) of p (the rest empty), its head
    void init(uint32_t n_cells, uint32_t swa, uint32_t h, const int32_t * p, uint32_t n) {
        size = n_cells; n_swa = swa; head = h;
        pos.assign(size, -1);
        used.clear(); sp.clear(); dirty.clear();
        is_dirty.assign(size, 0);
        for (uint32_t i = 0; i < n && i < size; ++i) {
            if (p[i] >= 0) put_(i, p[i]);
        }
        dirty.clear();
        std::fill(is_dirty.begin(), is_dirty.end(), 0);
    }

    int32_t pos_min() const { return sp.empty() ? -1 : sp.begin()->first; }
    int32_t pos_max() const { return sp.empty() ? -1 : sp.rbegin()->first; }
    uint32_t used_max_p1() const { return used.empty() ? 0 : *used.rbegin() + 1; }

    // llama_kv_cache::seq_rm(0, p0, p1): p1 < 0 is the end; head moves back to the first freed cell
    void seq_rm(int32_t p0, int32_t p1) {
        if (p0 < 0) p0 = 0;
        if (p1 < 0) p1 = INT32_MAX;
        if (p1 <= p0) return;
        std::vector<uint32_t> hit;
        for (auto it = sp.lower_bound({ p0, 0u }); it != sp.end() && it->first <= p1 - 1; ++it) hit.push_back(it->second);
        uint32_t new_head = size;
        for (const uint32_t i : hit) {
            clear_(i);
            new_head = std::min(new_head, i);
        }
        if (new_head != size && new_head < head) head = new_head;
    }

    // find_slot(cont = false): from head (from 0 when the cells before it hold enough free ones), wrapping, the first n
    // cells that are empty or out of the window of the sequence's newest position
    bool find_slot(int n, uint32_t * idxs) const {
        if ((uint32_t) n > size) return false;
        uint32_t head_cur = head;
        if (head_cur > (uint32_t) used.size() + 2u*(uint32_t) n) head_cur = 0;
        const int32_t p1 = pos_max() + 1;
        uint32_t n_tested = 0;
        int got = 0;
        while (true) {
            if (head_cur + 1 > size) {
                n_tested += size - head_cur;
                head_cur = 0;
                continue;
            }
            const uint32_t idx = head_cur++;
            n_tested++;
            if (pos[idx] < 0 || p1 - pos[idx] >= (int32_t) n_swa) idxs[got++] = idx;
            if (got == n) return true;
            if (n_tested >= size) return false;
        }
    }

    // apply_ubatch: the rows take their cells (overwriting windowed-out ones), positions below the newest overwritten one
    // are purged (the sequence stays contiguous), head moves past the last cell
    void apply(const uint32_t * idxs, const int32_t * ps, int n) {
        int32_t max_rm = -1;
        for (int i = 0; i < n; ++i) {
            if (pos[idxs[i]] >= 0) {
                max_rm = std::max(max_rm, pos[idxs[i]]);
                clear_(idxs[i]);
            }
            put_(idxs[i], ps[i]);
        }
        if (max_rm != -1 && pos_min() <= max_rm) seq_rm(pos_min(), max_rm + 1);
        head = idxs[n - 1] + 1;
    }

    // get_n_kv: the used span padded to 256, at most the cache
    int n_kv() const {
        const uint32_t pad = (used_max_p1() + 255u) / 256u * 256u;
        return (int) std::min(size, std::max(256u, pad));
    }

    // the changed cells since the last take, each once, with its position now: appended to out as (cell, pos) pairs
    template <typename D>
    int take(D * out) {
        int k = 0;
        for (const uint32_t i : dirty) {
            out[k].cell = (int32_t) i;
            out[k].pos  = pos[i];
            is_dirty[i] = 0;
            ++k;
        }
        dirty.clear();
        return k;
    }

private:
    void mark_(uint32_t i) {
        if (!is_dirty[i]) {
            is_dirty[i] = 1;
            dirty.push_back(i);
        }
    }
    void clear_(uint32_t i) {
        sp.erase({ pos[i], i });
        used.erase(i);
        pos[i] = -1;
        mark_(i);
    }
    void put_(uint32_t i, int32_t p) {
        pos[i] = p;
        used.insert(i);
        sp.insert({ p, i });
        mark_(i);
    }
};
