from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM


class ModelEvaluator():
    def __init__(self, model, tokenizer, device="cpu"):
        self.model = model.to(device)
        self.tokenizer = tokenizer 
        self.device = device

    def __call__(self, messages):
        if messages is not list:
            messages = [messages]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=1024
        ).detach().cpu()
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

def load_hf_model(model_name, device="cpu"):
    if 'gguf' in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            gguf_file=model_name.split('/')[1].lower()[:-5]+".gguf"
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            gguf_file=model_name.split('/')[1].lower()[:-5]+".gguf", 
            gpu_layers=0
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name
        )
    return ModelEvaluator(model, tokenizer, device)

def load_exp_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(
            model_name
        )
    return model, tokenizer
