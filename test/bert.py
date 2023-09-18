import torch
import welder
from transformers import BertTokenizer, BertModel

# Copy pasted from here https://huggingface.co/bert-base-uncased
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased").to(device="cuda:0")
model = torch.compile(
    model, backend="welder"
)  # This is the only line of code that we changed
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors="pt").to(device="cuda:0")
output = model(**encoded_input)


# torch compile
#   _TorchCompileInductorWrapper
# return torch._dynamo.optimize

#  _TorchCompileInductorWrapper:
# from torch._inductor.compile_fx import compile_fx
# return compile_fx(model_, inputs_, config_patches=self.config)

# compile_fx:
#   model_ = overrides.replace_fx(model_)
#   model_ = overrides.fuse_fx(model_, example_inputs_)
#   compile fw&bw:
#       compile_fx_inner


# compile_fx_inner
#   graphlowering

# graphlower
#   codegen
#        self.scheduler.codegen()


# Where codegen:
# torch._inductor.graph.py GraphLowering::compile_to_module
#   self.sheduler.codegen()
