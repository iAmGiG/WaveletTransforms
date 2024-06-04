from utils import print_model_summary, load_model, print_model_structure

model = load_model("__OGPyTorchModel__/pytorch_model.bin",
                   "__OGPyTorchModel__/config.json")

print_model_summary(model=model)
print_model_structure(model=model)
