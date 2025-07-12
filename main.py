import torch
import os
from train import train_model, parse_args

def main():
    args = parse_args()
    
    model, _, _, _ = train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        snr_db=args.snr_db,
        channel_type=args.channel_type,
        compression_ratio=args.compression_ratio,
        dataset=args.dataset
    )
    
    models_dir = './models'
    os.makedirs(models_dir, exist_ok=True)

    model_filename = f"td_jscc_{args.dataset}_snr{int(args.snr_db)}_{args.channel_type}_kn{args.compression_ratio:.4f}.pth"
    model_save_path = os.path.join(models_dir, model_filename)
    torch.save(model.state_dict(), model_save_path)
    print(f"\nModel saved to {model_save_path}")
    
    print("\n--- Training script finished successfully. ---")


if __name__ == "__main__":
    main()