from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,HfArgumentParser,TrainingArguments,pipeline, logging
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import torch
from datasets import Dataset
from trl import SFTTrainer
import pandas as pd
import os
import gc
from sklearn.utils import shuffle

MODELS = {
    "mistral": "mistralai/Mistral-7B-v0.3"
}


class DataAugmenter:
    def __init__(self, experiment_name, model_name, tokenizer_padding_side='right', add_special_tokens=False, 
                 max_length=1024, quantize_4bit=True, quantize_8bit=False, use_flash_attn=False, use_peft=True):
        self.experiment_name = experiment_name
        self.model_name = model_name
        self.tokenizer_padding_side = tokenizer_padding_side
        self.add_special_tokens = add_special_tokens
        self.max_length = max_length
        self.quantize_4bit = quantize_4bit
        self.quantize_8bit = quantize_8bit
        self.use_flash_attn = use_flash_attn
        self.use_peft = use_peft
        self.tokenizer = None
        self.model = None

        if model_name not in MODELS:
            raise ValueError(f"Model {model_name} not supported. Supported models: {list(MODELS.keys())}")
        
        self.base_model = MODELS[model_name]

        if quantize_4bit and quantize_8bit:
            raise ValueError("Only one of quantize_4bit and quantize_8bit can be True")
        
        self._init_tokenizer()
        self._init_model()
        
    def _init_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True, use_auth_token=True)
        self.tokenizer.padding_side = self.tokenizer_padding_side
        self.tokenizer.pad_token = self.tokenizer.unk_token # Check setting for models other than mistral
        
        if not self.add_special_tokens:
            self.tokenizer.add_eos_token = False
            self.tokenizer.add_bos_token = False
        self.tokenizer.model_max_length = self.max_length

        self.bos_token = self.tokenizer.bos_token
        self.eos_token = self.tokenizer.eos_token
        self.sep_token = "[SEP]"

    def _init_model(self):
        if self.quantize_4bit:
            self.bnb_config = BitsAndBytesConfig(
                load_in_4bit= True,
                bnb_4bit_quant_type= "nf4",
                bnb_4bit_compute_dtype= torch.bfloat16,
                bnb_4bit_use_double_quant= False,
            )
            
        elif self.quantize_8bit:
            self.bnb_config = BitsAndBytesConfig(
                load_in_8bit= True
            )

        else:
            self.bnb_config = None

        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=self.bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2" if self.use_flash_attn else None,
            trust_remote_code=True,
            use_auth_token=True
        )

        if self.quantize_4bit or self.quantize_8bit:
            self.model = prepare_model_for_kbit_training(self.model)
        
        if self.use_peft:
            self.lora_config = LoraConfig(
                lora_alpha=16,
                lora_dropout=0.1,
                r=64,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj"]
            )

            self.model = get_peft_model(self.model, self.lora_config)

        self.model.config.use_cache = False # silence the warnings
        self.model.config.pretraining_tp = 1
        self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads()
        self.model.resize_token_embeddings(len(self.tokenizer))

    def load_and_preprocess_dataset(self, filename:str) -> Dataset:
        """
        Load a dataset from a csv file, preprocess it and return it as a Hugging Face Dataset object

        Args:
            filename (str): Path to the csv file containing the dataset

        Returns:
            Dataset: Hugging Face Dataset object
        """
        df = pd.read_csv(filename)

        if not all(col in df.columns for col in ['text', 'label']):
            raise ValueError(f"File {filename} must have columns 'text' and 'label'")

        self._init_tokenizer()
        df['text'] = df.apply(lambda row: self.format_sample(row['label'], row['text']), axis=1)
        df = shuffle(df)
        dataset = Dataset.from_pandas(df)
        return dataset
    
    def train(self, train_dataset: Dataset, output_dir=None, optim="adamw_bnb_8bit", num_train_epochs=4,
              per_device_train_batch_size=4, gradient_accumulation_steps=1, save_steps=50, logging_steps=1,
              learning_rate=2.5e-5, weight_decay=0.001, fp16=False, bf16=False, max_grad_norm=0.3, max_steps=-1,
              warmup_ratio=0.03, group_by_length=True, gradient_checkpointing=True, lr_scheduler_type="constant", packing=False):
        """
        Train the model on the given dataset

        Args:
            train_dataset (Dataset): Hugging Face Dataset object containing the training data
            output_dir (str): Path to the directory where the trained model will be saved
            optim (str, optional): Optimizer to use. Defaults to "adamw_bnb_8bit".
            num_train_epochs (int, optional): Number of training epochs. Defaults to 4.
            per_device_train_batch_size (int, optional): Batch size per GPU. Defaults to 1.
            gradient_accumulation_steps (int, optional): Number of steps to accumulate gradients for. Defaults to 1.
            save_steps (int, optional): Number of steps after which to save the model. Defaults to 50.
            logging_steps (int, optional): Number of steps after which to log the training metrics. Defaults to 1.
            learning_rate (float, optional): Learning rate. Defaults to 2.5e-5.
            weight_decay (float, optional): Weight decay. Defaults to 0.001.
            fp16 (bool, optional): Whether to use FP16 training. Defaults to False.
            bf16 (bool, optional): Whether to use BF16 training. Defaults to False.
            max_grad_norm (float, optional): Maximum gradient norm. Defaults to 0.3.
            max_steps (int, optional): Maximum number of training steps. Defaults to -1.
            warmup_ratio (float, optional): Warmup ratio. Defaults to 0.03.
            group_by_length (bool, optional): Whether to group samples by length. Defaults to True.
            gradient_checkpointing (bool, optional): Whether to use gradient checkpointing. Defaults to True.
            lr_scheduler_type (str, optional): Learning rate scheduler type. Defaults to "constant".
            packing (bool, optional): Whether to use packing. Defaults to False.
        """
        if not self.tokenizer:
            self._init_tokenizer()

        if not self.model:
            self._init_model()

        training_arguments = TrainingArguments(
            output_dir=output_dir if output_dir else f'{self.experiment_name}_outputs',
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            optim=optim,
            save_steps=save_steps,
            logging_steps=logging_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            fp16=fp16,
            bf16=bf16,
            max_grad_norm=max_grad_norm,
            max_steps=max_steps,
            warmup_ratio=warmup_ratio,
            group_by_length=group_by_length,
            gradient_checkpointing=gradient_checkpointing,
            lr_scheduler_type=lr_scheduler_type
        )

        trainer = SFTTrainer(
            model=self.model,
            args=training_arguments,
            train_dataset=train_dataset,
            dataset_text_field='text',
            max_seq_length=self.tokenizer.model_max_length,
            tokenizer=self.tokenizer,
            packing=packing
        )

        trainer.train()

        self.model.save_pretrained(self.experiment_name)
        self.tokenizer.save_pretrained(self.experiment_name)

        self.model.push_to_hub(self.experiment_name, use_auth_token=True)
        self.tokenizer.push_to_hub(self.experiment_name, use_auth_token=True)

    def augment(self, labels, num_of_samples_per_label=10, min_length=10, max_length=100, temperature=1, top_k=30, top_p=0.90, repetition_penalty=1.5, tokenizer_side='right', generation_folder='generated_data'):
        self.model.use_cache = True
        old_tokenizer_padding_side = self.tokenizer_padding_side
        self.tokenizer.padding_side = tokenizer_side
        inputs = [self.format_sample(label) for label in labels]
        tokenized_prompts = self.tokenizer(inputs, padding=True, return_tensors='pt')
        model_op = self.model.generate(input_ids=tokenized_prompts['input_ids'].to('cuda'),
                                attention_mask=tokenized_prompts['attention_mask'].to('cuda'),
                                min_length=min_length,
                                max_length=max_length,
                                temperature=temperature,
                                top_k=top_k,
                                top_p=top_p,
                                repetition_penalty=repetition_penalty,
                                do_sample=True,
                                num_return_sequences=num_of_samples_per_label,
                                use_cache=True)
        generated_text = self.tokenizer.batch_decode(model_op, skip_special_tokens=True)
        self.tokenizer.padding_side = old_tokenizer_padding_side
        samples = [item.split(self.sep_token) for item in generated_text]
        samples = [(label.strip(), text.strip()) for (label, text) in samples]
        result_df = pd.DataFrame(samples, columns=['label', 'text'])

        os.makedirs(generation_folder, exist_ok=True)
        result_df.to_csv(f'{generation_folder}/{self.experiment_name}_augmented_data.csv', index=False)

        return result_df

    def format_sample(self, label, text = None, description=None):
        if not text:
            return f"{self.bos_token}{label} {self.sep_token}"
        
        return f"{self.bos_token}{label} {self.sep_token} {text}{self.eos_token}"
        

    def clean(self):
        self.model = None
        self.tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()