#   out = self.model(inp_ids, past_key_values=past_keys)

# Replace it with something that also passes the bias:
#   out = self.model(inp_ids, past_key_values=past_keys, bias=bias_tensor)
# where `bias_tensor` comes from `_get_mem_past`.

# Example (if your model’s forward signature is `forward(input_ids, past_key_values=None,
#                                            bias=None)`):
out = self.model(input_ids=inp_ids,
                 past_key_values=past_keys,
                 bias=self._get_mem_past(mem_embs)[2])   # only the bias part we need
