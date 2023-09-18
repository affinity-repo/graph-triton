import welder
import torch
from absl import app, flags
from transformers import BertTokenizer, BertModel

model_list = {
    "bert-case-uncased": [],
}

FLAGS = flags.FLAGS
flags.DEFINE_string("model", "bert-base-uncased", "")
flags.DEFINE_string("output", "output_triton_code.py", "")
flags.DEFINE_string("device", "cuda:0", "")


def bert_base_uncased():
    # Copy pasted from here https://huggingface.co/bert-base-uncased
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased").to(device="cuda:0")
    model = torch.compile(model, backend="welder")
    text = "Replace me by any text you'd like."
    encoded_input = tokenizer(text, return_tensors="pt").to(device="cuda:0")
    output = model(**encoded_input)
    return output


def main(argv):
    if FLAGS.model == "bert-base-uncased":
        pass


if __name__ == "__main__":
    app.run(main)

# Supported patterns more than torch.inductor
# [ ] Gemm + Pointwise + Gemm
# [ ] Conv + Pointwise
# [ ] Gemm + Pointwise
# [ ] Pointwise + Gemm/Conv

# Workflow
# 1. Setup model + input tensor
# 2. Generate Schedule + Read Welder Schedule
# 3. Generate Python/Triton Code
# 4. Compile, Run and Get Pefromance
