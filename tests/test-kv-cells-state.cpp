#include "../src/llama-kv-cells.h"

#include <cstdio>
#include <vector>

int main() {
    llama_kv_cells cells;
    cells.resize(4);

    cells.pos_set(0, 10);
    cells.seq_add(0, 0);

    llama_kv_cell_ext ext;
    ext.tok = 42;
    cells.ext_set(0, ext);

    const std::vector<uint32_t> idxs = { 0, 2 };
    const auto state = cells.state_get(idxs);

    cells.seq_rm(0, 0);
    cells.pos_set(2, 30);
    cells.seq_add(2, 1);

    cells.state_set(idxs, state);

    if (cells.is_empty(0) || cells.pos_get(0) != 10 || !cells.seq_has(0, 0) ||
            cells.ext_get(0).tok != 42 || !cells.is_empty(2) || cells.get_used() != 1) {
        std::fprintf(stderr, "test-kv-cells-state: restored state differs\n");
        return 1;
    }

    std::printf("test-kv-cells-state: all checks passed\n");
    return 0;
}
