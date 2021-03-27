from chatbot import Chatbot

bot = Chatbot("Test")
bot.train("data/intents.json")
bot.save("models")
print(bot.model)
print(bot._intents)

# bot = Chatbot()
# bot.load("models/Test")
# print(bot.name)
# print(bot.model)
# print(bot._intents)

# bot = Chatbot.load_chatbot("models/Test")
# print(bot.name)
# print(bot.model)
# print(bot._intents)

# bot = Chatbot.load_chatbot("models/Test")
# bot.greet()
# bot.prompt_and_respond()

# bot = Chatbot()
# print(Chatbot().__dict__)