// split-K の部分和を z の小さい順に足す。part は [splits, len](vec4)、out は [len]。

@group(0) @binding(0) var<storage, read_write> part: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> out: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> dims: array<u32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let len = dims[0];
    let splits = dims[1];
    let i = gid.x + gid.y * nwg.x * 256u;
    if (i >= len) {
        return;
    }
    var acc = part[i];
    for (var z = 1u; z < splits; z++) {
        acc += part[z * len + i];
    }
    out[i] = acc;
}
