from discord.ext import commands
from pprint import pprint

from MachineLearning.classify import predict

bad_words = ["sexy", "hot", "sex"]

class Utilities(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.command(brief="Delete a conversation")
    async def delete(self, ctx, start=None, end=None):
        """Command to delete conversation"""
        if ctx.message.author.bot:
            return

        if not start or not end:
            await ctx.send("Insufficient arguments!\n Arguements: <start ID> <end ID>")
            return

        channel = ctx.channel
        try:
            start_message = await channel.fetch_message(start)
            end_message = await channel.fetch_message(end)
        except:
            await ctx.send("Can not fetch message!")
            return

        try:
            raw_messages = await channel.history(
                before=end_message.created_at,
                after=start_message.created_at,
                oldest_first=True,
            ).flatten()
        except:
            await ctx.send("Can not get messages in that time range!")
            return

        raw_messages = [start_message, *raw_messages, end_message]

        for message in raw_messages:
            await message.delete()

    @commands.Cog.listener()
    async def on_message(self, message):
        if message.author.bot:
            return

        prediction = predict(message.content)
        prediction = 0

        for word in bad_words:
            if word in message.content:
                prediction = 1

        if prediction == 1:
            await message.channel.send("Your message might have been inappropriate.") 


def setup(bot):
    bot.add_cog(Utilities(bot))
