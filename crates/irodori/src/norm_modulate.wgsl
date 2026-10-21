// adaLN: 行ごとの RMSNorm(重みなし)に変調を掛ける。out = x * inv * (scale + 1) + shift。
// x は [rows, d]、shift / scale は [mod_batches, d](1 なら全行で共通、そうでなければ行の属するバッチ row / seq の行)。
// 1 ワークグループが 1 行を受け持ち、二乗和は共有メモリで足し合わせる。

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> shift: array<f32>;
@group(0) @binding(2) var<storage, read_write> scale: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<storage, read_write> dims: array<u32>;

var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let rows = dims[0];
    let d = dims[1];
    let seq = dims[2];
    let mod_batches = dims[3];
    let eps = bitcast<f32>(dims[4]);
    let tid = lid.x;
    // ワークグループ全体が同じ行なので、範囲外でも return せずに書き込みだけ止める(バリアを一様に保つ)
    let row = wid.x + wid.y * nwg.x;
    let valid = row < rows;
    let base = min(row, rows - 1u) * d;
    var part = 0.0;
    for (var i = tid; i < d; i += 256u) {
        let v = x[base + i];
        part += v * v;
    }
    red[tid] = part;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st >>= 1u) {
        if (tid < st) {
            red[tid] += red[tid + st];
        }
        workgroupBarrier();
    }
    let inv = 1.0 / sqrt(red[0] / f32(d) + eps);
    var mb = 0u;
    if (mod_batches > 1u) {
        mb = (row / seq) * d;
    }
    if (valid) {
        for (var i = tid; i < d; i += 256u) {
            out[base + i] = x[base + i] * inv * (scale[mb + i] + 1.0) + shift[mb + i];
        }
    }
}
