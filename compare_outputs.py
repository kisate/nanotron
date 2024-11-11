import torch
hf = torch.load(
    f"img_emb_hf_{26}_after_mlp.pt",
).detach().cpu()
nano = torch.load(
    f"img_emb_nano_{26}_after_mlp.pt",
).detach().cpu()


def compare_outputs(hf, nano):
    print(
        hf.shape, nano.shape
    )
    
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