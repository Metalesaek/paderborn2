import threading
from django.conf import settings

from pathlib import Path
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
import openai
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from django.templatetags.static import static
from .views import *



'''
#llm = ChatOpenAI(model_name="gpt-4-0613", openai_api_key="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")
#embeddings = OpenAIEmbeddings()
#url = static('chroma2') 
#docsearch = Chroma(
 #  persist_directory= url, embedding_function=embeddings
  #)

#memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
#docsearch=settings.get('DOCSEARCH')
#retriever = docsearch.as_retriever()
# Create the multipurpose chain
qachat = ConversationalRetrievalChain.from_llm(
            llm=llm,
            condense_question_llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo'),
            memory=memory,
            retriever=retriever, 
            return_source_documents=True
        )

def run_my_function():
    """
    This function starts a thread that runs the `my_function` function once.
    """
    thread = threading.Thread(target=qachat)
    thread.daemon = True
    thread.start()

if __name__ == '__main__':
    run_my_function()


    '''