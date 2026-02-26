import tensorrt as trt

ONNX_PATH = "models/nitec_rs18_e20_opset17_legacy.onnx"
ENGINE_PATH = "models/nitec_rs18_e20_fp32.engine"

logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

with open(ONNX_PATH, "rb") as f:
    ok = parser.parse(f.read())

if not ok:
    print("❌ ONNX parse failed. Errors:")
    for i in range(parser.num_errors):
        print(parser.get_error(i))
    raise RuntimeError("Still failing. Then we’ll inspect ONNX initializer dtype/format.")

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

profile = builder.create_optimization_profile()
profile.set_shape("images", (1,3,224,224), (8,3,224,224), (32,3,224,224))
config.add_optimization_profile(profile)

serialized_engine = builder.build_serialized_network(network, config)
if serialized_engine is None:
    raise RuntimeError("Engine build failed.")

with open(ENGINE_PATH, "wb") as f:
    f.write(serialized_engine)

print("✅ Engine saved:", ENGINE_PATH)