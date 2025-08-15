from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_cohere import CohereEmbeddings
from dotenv import load_dotenv
import os
from enum import Enum


try:
    from backend.src.utils.logger_utils import setup_logging
    
    # Load environment variables from backend/.env file
    backend_env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
    load_dotenv(backend_env_path, override=True)

    # Set up logging
    logger = setup_logging(__name__)
    
    # Log environment configuration status
    logger.debug("🔧 AI Models module loaded")
    logger.debug(f"🔧 ONLY_CHEAP_LLM: {os.getenv('ONLY_CHEAP_LLM', 'not set')}")
    logger.debug(f"🔧 Azure endpoint configured: {'Yes' if os.getenv('AZURE_OPENAI_API_ENDPOINT') else 'No'}")
    logger.debug(f"🔧 API key configured: {'Yes' if os.getenv('AZURE_OPENAI_API_KEY') else 'No'}")
    logger.debug(f"🔧 Cohere API key configured: {'Yes' if os.getenv('AZURE_COHERE_EMBEDDING_API_KEY') else 'No'}")
    logger.debug(f"🔧 Cohere endpoint configured: {'Yes' if os.getenv('AZURE_COHERE_EMBEDDING_API_ENDPOINT') else 'No'}")
    
except:
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())
    # Load environment variables from backend/.env file as fallback
    backend_env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
    load_dotenv(backend_env_path, override=True)



# Global variables to store LLM instances
_intelligent_llm = None
_cheap_llm = None

#define enum for LLM types
class LLMType(Enum):
    INTELLIGENT = "intelligent"
    CHEAP = "cheap"

def _initialize_llm(intelligent_or_cheap: LLMType) -> AzureChatOpenAI:
    """
    Initialize and return an instance of AzureChatOpenAI LLM.

    Args:
        intelligent_or_cheap (str): Specify whether to initialize the 'intelligent' or 'cheap' LLM.

    Returns:
        AzureChatOpenAI: An instance of the AzureChatOpenAI LLM.
    """
    logger.info(f"🔧 Initializing LLM: {intelligent_or_cheap.value}")
    
    try:
        if intelligent_or_cheap == LLMType.INTELLIGENT:
            logger.debug(f"🎯 Creating INTELLIGENT LLM with deployment: {os.getenv('AZURE_OPENAI_LLM_DEPLOYMENT_NAME_INTELLIGENT')}")
            logger.debug(f"🎯 Model: {os.getenv('AZURE_OPENAI_LLM_MODEL_NAME_INTELLIGENT')}")
            logger.debug(f"🎯 Endpoint: {os.getenv('AZURE_OPENAI_API_ENDPOINT')}")
            logger.debug(f"🎯 API Version: {os.getenv('AZURE_OPENAI_API_VERSION')}")
            
            llm = AzureChatOpenAI(
                deployment_name=os.getenv("AZURE_OPENAI_LLM_DEPLOYMENT_NAME_INTELLIGENT"),
                model=os.getenv("AZURE_OPENAI_LLM_MODEL_NAME_INTELLIGENT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                azure_endpoint=os.getenv("AZURE_OPENAI_API_ENDPOINT"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                temperature=os.getenv("TEMPERATURE_GPT"),
                #use_responses_api=True,
                #reasoning_effort="low"
            )
            logger.info("✅ INTELLIGENT LLM initialized successfully")
            return llm
            
        elif intelligent_or_cheap == LLMType.CHEAP:
            logger.debug(f"💰 Creating CHEAP LLM with deployment: {os.getenv('AZURE_OPENAI_LLM_DEPLOYMENT_NAME_CHEAP')}")
            logger.debug(f"💰 Model: {os.getenv('AZURE_OPENAI_LLM_MODEL_NAME_CHEAP')}")
            logger.debug(f"💰 Endpoint: {os.getenv('AZURE_OPENAI_API_ENDPOINT')}")
            logger.debug(f"💰 API Version: {os.getenv('AZURE_OPENAI_API_VERSION')}")
            
            llm = AzureChatOpenAI(
                deployment_name=os.getenv("AZURE_OPENAI_LLM_DEPLOYMENT_NAME_CHEAP"),
                model=os.getenv("AZURE_OPENAI_LLM_MODEL_NAME_CHEAP"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                azure_endpoint=os.getenv("AZURE_OPENAI_API_ENDPOINT"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                temperature=os.getenv("TEMPERATURE_GPT"),
                #use_responses_api=True,
                #reasoning_effort="low"
            )
            logger.info("✅ CHEAP LLM initialized successfully")
            return llm
        else:
            logger.error(f"❌ Invalid LLM type: {intelligent_or_cheap}")
            raise ValueError(f"Invalid LLM type: {intelligent_or_cheap}")
    except Exception as e:
        logger.error(f"❌ Failed to initialize LLM ({intelligent_or_cheap.value}): {e}")
        logger.error(f"❌ Error type: {type(e).__name__}")
        raise

def get_llm(intelligent_or_cheap: LLMType) -> AzureChatOpenAI:
    """
    Get the initialized LLM instance. If not initialized, initialize it first.

    Args:
        intelligent_or_cheap (LLMType): Specify whether to get the 'intelligent' or 'cheap' LLM.

    Returns:
        AzureChatOpenAI: An instance of the AzureChatOpenAI LLM.
    """
    global _intelligent_llm, _cheap_llm

    logger.info(f"🔍 Requesting LLM: {intelligent_or_cheap.value}")

    # Check if ONLY_CHEAP_LLM is set
    if os.getenv("ONLY_CHEAP_LLM") == "true":
        logger.info("🔄 ONLY_CHEAP_LLM=true, forcing CHEAP LLM")
        intelligent_or_cheap = LLMType.CHEAP

    if intelligent_or_cheap == LLMType.INTELLIGENT:
        if _intelligent_llm is None:
            logger.info("🚀 INTELLIGENT LLM not cached, initializing...")
            _intelligent_llm = _initialize_llm(LLMType.INTELLIGENT)
        else:
            logger.info("♻️ Using cached INTELLIGENT LLM")
        logger.info("✅ Returning INTELLIGENT LLM")
        return _intelligent_llm
        
    elif intelligent_or_cheap == LLMType.CHEAP:
        if _cheap_llm is None:
            logger.info("🚀 CHEAP LLM not cached, initializing...")
            _cheap_llm = _initialize_llm(LLMType.CHEAP)
        else:
            logger.info("♻️ Using cached CHEAP LLM")
        logger.info("✅ Returning CHEAP LLM")
        return _cheap_llm
        
    else:
        logger.error(f"❌ Invalid LLM type in get_llm: {intelligent_or_cheap}")
        raise ValueError(f"Invalid LLM type: {intelligent_or_cheap}")

def get_embedding_model():
    return CohereEmbeddings(
        model="embed-v-4-0",
        cohere_api_key=os.getenv("AZURE_COHERE_EMBEDDING_API_KEY"),
        base_url=os.getenv("AZURE_COHERE_EMBEDDING_API_ENDPOINT"),
        request_timeout=1000000,  # 30 second timeout per request
        max_retries=2,       # Maximum 2 retries (3 total attempts)
    )
    
    
def test_llm():
    logger.info("🧪 Starting LLM test")
    try:
        llm = get_llm(LLMType.CHEAP)
        logger.info(f"🧪 LLM instance: {llm}")
        logger.info("🧪 Invoking test message...")
        response = llm.invoke("Hello, how are you?")
        logger.info(f"🧪 LLM response: {response}")
        logger.info("✅ LLM test completed successfully")
    except Exception as e:
        logger.error(f"❌ LLM test failed: {e}")
        raise


def test_embedding():
    logger.info("🧪 Starting embedding test")
    try:
        embedding = get_embedding_model()
        logger.info(f"🧪 Embedding model: {embedding}")
        logger.info("🧪 Computing test embedding...")
        result = embedding.embed_query("Hello, how are you?")
        logger.info(f"🧪 Embedding result length: {len(result) if result else 'None'}")
        logger.info("✅ Embedding test completed successfully")
    except Exception as e:
        logger.error(f"❌ Embedding test failed: {e}")
        raise
    
if __name__ == "__main__":
    #test_llm()
    test_embedding()