from chatbot import Chatbot

bot = Chatbot()
bot.train("data/intents.json")
bot.save("models")