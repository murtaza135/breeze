from chatbot import Chatbot

# bot = Chatbot("Test")
# bot.train("data/intents.json")
# bot.save("models")
# print(bot.model)
# print(bot._intents)
# print(bot.actions)

# bot = Chatbot()
# bot.load("models/Test")
# print(bot.name)
# print(bot.model)
# print(bot._intents)
# print(bot.actions)

# bot = Chatbot.load_chatbot("models/Test")
# print(bot.name)
# print(bot.model)
# print(bot._intents)

def func(message):
    print(message)

bot = Chatbot.load_chatbot("models/Test")
bot.map_function_to_tag("greeting", func, "Hello World")
bot.greet()
bot.prompt_and_respond()