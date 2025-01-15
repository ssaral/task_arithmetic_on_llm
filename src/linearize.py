import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from functorch import jvp, make_functional_with_buffers

class LinearizedModel(nn.Module):
    """Creates a linearized version of a nn.Module.

    Args:
        model (nn.Module): The model to linearize. The trainable parameters of
            the linearized model will be initialized to the parameters of this
            model.
        init_model (nn.Module): A model of the same type as `model` containing
            the parameters around which the model is initialized. If not
            provided, `model` is used as the initialization model.
    """
    def __init__(self, model: nn.Module, init_model: nn.Module = None) -> None:
        super().__init__()
        if init_model is None:
            init_model = model

        func0, params0, self.buffers0 = make_functional_with_buffers(
            init_model.eval(), disable_autograd_tracking=True
        )
        self.func0 = lambda params, x: func0(params, self.buffers0, x)

        _, params, _ = make_functional_with_buffers(
            model, disable_autograd_tracking=True
        )

        self.params = nn.ParameterList(params)
        self.params0 = nn.ParameterList(params0)
        self._model_name = model.__class__.__name__

        # The initial parameters are not trainable.
        for p in self.params0:
            p.requires_grad = False

        # The params are trainable.
        for p in self.params:
            p.requires_grad = True

    def forward(self, x) -> torch.Tensor:
        """Computes the linearized model output using a first-order Taylor decomposition."""
        dparams = [p - p0 for p, p0 in zip(self.params, self.params0)]
        out, dp = jvp(
            lambda param: self.func0(param, x),
            (tuple(self.params0),),
            (tuple(dparams),),
        )

        return out.logits + dp.logits
    
    @classmethod
    def load(cls, filename, model):
        """Loads a linearized language model."""

        print(f"Loading language model from {filename}")
        state_dict = torch.load(filename, map_location="cpu")

        # Extract model name and initialize the model
        linearized_lm = cls(model)

        # Load state dict
        linearized_lm.load_state_dict(state_dict)
        return linearized_lm

class LinearizedLM(nn.Module):
    """Creates a linearized version of a language model (e.g., GPT-2)."""
    # def __init__(self, model):#, init_model_name=None):
    #     super().__init__()
        
    #     # Load models
    #     # self.model = AutoModelForCausalLM.from_pretrained(model_name)
    #     # if init_model_name:
    #     #     self.init_model = AutoModelForCausalLM.from_pretrained(init_model_name)
    #     # else:
    #     #     self.init_model = AutoModelForCausalLM.from_pretrained(model_name)

    #     # Create linearized model
    #     # self.linearized_model = LinearizedModel(model=self.model, init_model=self.init_model)
    #     self.linearized_model = LinearizedModel(model)
    def __init__(self, model: nn.Module, init_model: nn.Module = None) -> None:
        super().__init__()
        if init_model is None:
            init_model = model

        lm = LinearizedModel(model)
        self.linearized_model = lm
        # print(model.state_dict().keys())
        

    # def forward(self, input_ids, attention_mask=None, labels=None):
    #     """Forward pass through the linearized language model."""
    #     inputs = {
    #         "input_ids": input_ids,
    #         "attention_mask": attention_mask,
    #     }
    #     outputs = self.linearized_model(inputs)
    #     # outputs = self.linearized_model(input_ids)

    #     print("Labels", labels)
    #     print("\n")

    #     print("Outputs", outputs)

    #     # print("Labels shape", labels.shape)
    #     print("Outputs shape", outputs.shape)

    #     # If labels are provided, compute loss
    #     if labels is not None:
    #         loss_fct = nn.CrossEntropyLoss()
    #         shift_logits = outputs[..., 1:].contiguous()
    #         shift_labels = labels[..., 1:].contiguous()
    #         loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    #         return loss

    #     return outputs


    # @classmethod
    # def load(cls, filename, model):
    #     """Loads a linearized language model."""
    #     # print(f"Loading language model from {filename}")
    #     # state_dict = torch.load(filename, map_location="cpu")

    #     # # Extract model name and initialize the model
    #     print("Model is set to GPT2. Please change the model name in linearize.py (line 137). This manual intervention will be resolved.")
    #     # model_name = "gpt2"
    #     # linearized_lm = cls(model_name=model_name)

    #     # # Load state dict
    #     # linearized_lm.load_state_dict(state_dict)
    #     # return linearized_lm
    
    #     print(f"Loading language model from {filename}")
    #     state_dict = torch.load(filename, map_location="cpu")
    #     print("State dicts loaded")
    #     print("linearizing the model")
    #     # Extract model name and initialize the model
    #     linearized_lm = cls(model)

    #     print("load state dict")
    #     # Load state dict
    #     linearized_lm.load_state_dict(state_dict)
    #     return linearized_lm
