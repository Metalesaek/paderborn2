from django.apps import AppConfig


class SmallChatbotConfig(AppConfig):
	default_auto_field = 'django.db.models.BigAutoField'
	name = 'small_chatbot'
	'''
	def ready(self):
		# Import the signal to trigger it
		from small_chatbot.signals import service_started

		# Send the signal to execute the connected function
		service_started.send(sender=self.__class__)
'''