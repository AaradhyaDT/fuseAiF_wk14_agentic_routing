from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, get_peft_model_state_dict, set_peft_model_state_dict

def train_slm(model_name, train_df, val_df, epochs=SLM_EPOCHS, lr=SLM_LR, batch_size=SLM_BATCH_SIZE):
    print(f"\n{'='*60}\nTraining: {model_name}\n{'='*60}")
    set_seed()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, device_map={"": 0}
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_ds = SLMIntentDataset(train_df["instruction"], train_df["category"], tokenizer)
    val_ds = SLMIntentDataset(val_df["instruction"], val_df["category"], tokenizer)
    collate_fn = lambda b: slm_collate(b, tokenizer.pad_token_id)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps
    )

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_adapter_state = None

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            optimizer.zero_grad()
            out = model(**batch)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_train_loss += loss.item()
        avg_train_loss = epoch_train_loss / len(train_loader)

        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                out = model(**batch)
                epoch_val_loss += out.loss.item()
        avg_val_loss = epoch_val_loss / len(val_loader)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        print(f"Epoch {epoch+1}/{epochs} | train_loss={avg_train_loss:.4f} | val_loss={avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_adapter_state = get_peft_model_state_dict(model)
            best_adapter_state = {k: v.cpu().clone() for k, v in best_adapter_state.items()}

    # Reload best LoRA weights — PEFT-native extraction, immune to bitsandbytes
    # quant-buffer key collisions (.absmax/.quant_map/.nested_*/.quant_state.*)
    # that a "lora" in k substring match can pull in on a 4-bit base.
    set_peft_model_state_dict(model, best_adapter_state)

    return model, tokenizer, history
