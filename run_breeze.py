from chatbot import Chatbot
import webbrowser

# bot = Chatbot("data/intents.json", "Test")
# bot.train()
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

def youtube():
    url = "https://www.youtube.com/"
    webbrowser.get().open(url)

bot = Chatbot.load_chatbot("models/Test")
bot.map_function_to_tag("youtube", youtube)
bot.speak("Hello, how can I help you?")
for i in range(3):
    bot.prompt_and_respond()
    print("====================================================")
