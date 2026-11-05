import torch


class Benchmark:
    @staticmethod
    def get_model_and_inputs(model_name, device):
        raise NotImplementedError("get_model_and_inputs() not implemented")


class WhisperBenchmark(Benchmark):
    SAMPLE_RATE = 16000
    DURATION = 30.0

    @staticmethod
    def get_model_and_inputs(model_name, device):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        processor = WhisperProcessor.from_pretrained(model_name)
        model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
        model.config.forced_decoder_ids = None
        model.generation_config.do_sample = False
        model.generation_config.temperature = 0.0
        num_samples = int(WhisperBenchmark.DURATION * WhisperBenchmark.SAMPLE_RATE)
        audio = torch.randn(num_samples) * 0.1
        inputs = dict(
            processor(
                audio,
                sampling_rate=WhisperBenchmark.SAMPLE_RATE,
                return_tensors="pt",
            )
        )
        inputs["input_features"] = inputs["input_features"].to(device)
        inputs["decoder_input_ids"] = torch.tensor(
            [[model.config.decoder_start_token_id]], device=device
        )
        return model, inputs


class TextGenerationBenchmark(Benchmark):
    INPUT_LENGTH = 1000
    OUTPUT_LENGTH = 2000

    @staticmethod
    def get_model_and_inputs(model_name, device):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        model.eval()
        model.generation_config.do_sample = False
        model.generation_config.use_cache = True
        model.generation_config.cache_implementation = "static"
        model.generation_config.max_new_tokens = TextGenerationBenchmark.OUTPUT_LENGTH
        model.generation_config.pad_token_id = tokenizer.eos_token_id
        model.generation_config.temperature = 0.0
        inputs = {
            "input_ids": torch.randint(
                low=0,
                high=tokenizer.vocab_size,
                size=(1, TextGenerationBenchmark.INPUT_LENGTH),
                device=device,
                dtype=torch.long,
            )
        }
        return model, inputs


HF_LLM_MODELS = {
    "meta-llama/Llama-3.2-1B": TextGenerationBenchmark,
    "google/gemma-2-2b": TextGenerationBenchmark,
    "google/gemma-3-4b-it": TextGenerationBenchmark,
    "openai/whisper-tiny": WhisperBenchmark,
    "Qwen/Qwen3-0.6B": TextGenerationBenchmark,
    "mistralai/Mistral-7B-Instruct-v0.3": TextGenerationBenchmark,
    "openai/gpt-oss-20b": TextGenerationBenchmark,
}
