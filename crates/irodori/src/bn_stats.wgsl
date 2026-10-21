// 透かしの BatchNorm(学習モード)の統計: チャネルごとに有効な位置の平均と偏分散を取り、
// 掛ける係数 a = w * rsqrt(var + eps) と足す値 shift = b - mean * a を書く(ab[2c], ab[2c + 1])。
// y の無効な位置は 0。1 ワークグループが 1 チャネルを受け持ち、平均を出してから偏差の二乗和を取る(二段で計算する原典と同じ)。

@group(0) @binding(0) var<storage, read_write> y: array<f32>;
@group(0) @binding(1) var<storage, read_write> bn_w: array<f32>;
@group(0) @binding(2) var<storage, read_write> bn_b: array<f32>;
@group(0) @binding(3) var<storage, read_write> ab: array<f32>;
// [n, wp, w, ncols, n_valid(f32), eps(f32)]
@group(0) @binding(4) var<storage, read_write> dims: array<u32>;

var<workgroup> part: array<f32, 256>;

fn valid(n: u32) -> bool {
    return n < dims[3] && n % dims[1] < dims[2];
}

fn reduce(lid: u32, v: f32) -> f32 {
    part[lid] = v;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (lid < s) {
            part[lid] += part[lid + s];
        }
        workgroupBarrier();
    }
    let total = part[0];
    workgroupBarrier();
    return total;
}

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let n = dims[0];
    let n_valid = bitcast<f32>(dims[4]);
    let eps = bitcast<f32>(dims[5]);
    let c = wid.x;
    let base = c * n;
    var s = 0.0;
    for (var p = lid.x; p < n; p += 256u) {
        s += y[base + p];
    }
    let mean = reduce(lid.x, s) / n_valid;
    var q = 0.0;
    for (var p = lid.x; p < n; p += 256u) {
        if (valid(p)) {
            let d = y[base + p] - mean;
            q += d * d;
        }
    }
    let var_ = reduce(lid.x, q) / n_valid;
    if (lid.x == 0u) {
        let a = bn_w[c] / sqrt(var_ + eps);
        ab[2u * c] = a;
        ab[2u * c + 1u] = bn_b[c] - mean * a;
    }
}
