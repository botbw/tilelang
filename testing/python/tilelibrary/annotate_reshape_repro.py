

def decompose_col_major(index_1d: int, index_nd: list[int]) -> list[int]:
    ori_idx = index_1d
    res = []
    for x in index_nd:
        res.append(index_1d % x)
        index_1d //= x
    assert index_1d == 0, f"{index_1d=} {ori_idx=} {res=}"
    return res

def global_mapping_row_major(i, j):
    # E global layout: (((_8,_2,_4),2),((_2,_2,_4),2)):(((_16,_2,_256),_1024),((_1,_128,_4),2048))
    # (128, 32)
    i_decomposed = decompose_col_major(i, [8, 2, 4, 2])
    j_decomposed = decompose_col_major(j, [2, 2, 4, 2])
    stride_i = [16, 2, 256, 1024]
    stride_j = [1, 128, 4, 2048]
    i_offset = sum(i_decomposed[k] * stride_i[k] for k in range(len(i_decomposed)))
    j_offset = sum(j_decomposed[k] * stride_j[k] for k in range(len(j_decomposed)))

    mem_offset = i_offset + j_offset
    return mem_offset

def shared_mapping_row_major(i, j):
    # (128, 16)
    # E shared layout: ((_8,_2,_4, 2),(_2,_2,_4)):((_16,_2,_256, 1024),(_1,_128,_4))
    i_decomposed = decompose_col_major(i, [8, 2, 4, 2])
    j_decomposed = decompose_col_major(j, [2, 2, 4])
    stride_i = [16, 2, 256, 1024]
    stride_j = [1, 128, 4]
    i_offset = sum(i_decomposed[k] * stride_i[k] for k in range(len(i_decomposed)))
    j_offset = sum(j_decomposed[k] * stride_j[k] for k in range(len(j_decomposed)))

    mem_offset = i_offset + j_offset

    return mem_offset


for i in range(127):
    for j in range(31):
        print(f"{global_mapping_row_major(i, j):4}", end=" ")
    print()