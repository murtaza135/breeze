from chatbot import Chatbot

# bot = Chatbot()
# bot.train("data/intents.json")
# bot.save("models")

# bot = Chatbot()
# bot.load("models/Test")
# print(bot.name)
# print(bot.model)
# print(bot._intents)

bot = Chatbot.load_model("models/Test")
print(bot.name)
print(bot.model)
print(bot._intents)