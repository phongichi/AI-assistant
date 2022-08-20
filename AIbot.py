from transformers import BlenderbotSmallTokenizer, BlenderbotSmallForConditionalGeneration
import os
from speak_module import speak
from listen_module import listen
from welcome import greet




class BlenderBot:
    def __init__(
        self,
        model_name: str ='facebook/blenderbot_small-90M',
    ):
        if not os.path.exists('./models/blenderbot'):
            BlenderbotSmallForConditionalGeneration.from_pretrained(model_name).save_pretrained('./models/blenderbot')
            BlenderbotSmallTokenizer.from_pretrained(model_name).save_pretrained('./models/blenderbot')

        self.model = BlenderbotSmallForConditionalGeneration.from_pretrained('./models/blenderbot')
        self.tokenizer = BlenderbotSmallTokenizer.from_pretrained('./models/blenderbot')

    def __call__(self, inputs: str) -> str:
        inputs_tokenized = self.tokenizer(inputs, return_tensors='pt')
        reply_ids = self.model.generate(**inputs_tokenized)
        reply = self.tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]

        return reply

    def run(self):
            greet(listen())
            while True:
                user_input = listen()
                speak(self(user_input))


