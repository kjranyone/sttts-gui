// 透かしの BatchNorm の適用: y [C, n] を a * y + shift にして、縁取り付きの平らな配列 [C, buf_len] に置く
// (出力の位置 p は平らな配列の wp + 1 + p。無効な位置と縁取りは 0)。

@group(0) @binding(0) var<storage, read_write> y: array<f32>;
@group(0) @binding(1) var<storage, read_write> ab: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
// [n, wp, w, ncols, buf_len]
@group(0) @binding(3) var<storage, read_write> dims: array<u32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = dims[0];
    let wp = dims[1];
    let buf_len = dims[4];
    let q = gid.x;
    let c = gid.y;
    if (q >= buf_len) {
        return;
    }
    var v = 0.0;
    if (q > wp) {
        let p = q - wp - 1u;
        if (p < n && p < dims[3] && p % wp < dims[2]) {
            v = y[c * n + p] * ab[2u * c] + ab[2u * c + 1u];
        }
    }
    out[c * buf_len + q] = v;
}
