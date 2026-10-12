;; qwen-cpu-cold-experts (layer 0; run by gp-cpu-join): 2 records, 0 slots
(program qwen-cpu-cold-experts
  (env )
  (kq-quant %17:f32[1x2048] 1 2048 %185:i8[1x2048] %186:f32[1x64] %187:f32[1x128])
  (kq-moe %185:i8[1x2048] %186:f32[1x64] %187:f32[1x128] %19:i32[17] %18:f32[8] 1 8 256 %188:i64[12] 0 2048 512 %189:u8[146528] %24:f32[1x2048] %19:i32[17]+32)
)
