import torch
layer = 31
hf = torch.load(
    f"hf_output_text_llama.pt",
).last_hidden_state.detach().cpu()
nano = torch.load(
    f"nano_output_text_llama.pt",
).detach().cpu().transpose(0, 1)


def compare_outputs(hf, nano):
    print(
        hf.shape, nano.shape
    )

    if hf.shape != nano.shape:
        print("Shapes are different")
        return
    
    print(
        f"Max difference between outputs: {torch.max(torch.abs(hf - nano))}"
    )
    print(
        f"Mean difference between outputs: {torch.mean(torch.abs(hf - nano))}"
    )
    print(
        f"Relative difference between outputs: {torch.mean(torch.abs(hf - nano) / (torch.abs(hf) + 1e-6))}"
    )

    # print(
    #     hf[0, 0, :15], nano[0, 0, :15]
    # )

compare_outputs(hf, nano)



# a = torch.load(f"img_emb_hf_{0}_attention_mask.pt")
# b = torch.load(f"img_emb_hf_{0}_args.pt")
# c = torch.load(f"img_emb_nano_{0}_qkv.pt")

# print(a)
# print(b)
# print(c.shape)