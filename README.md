# Functions, Tools and Agents with LangChain

[![DeepLearning.AI](https://img.shields.io/badge/DeepLearning.AI-Short%20Course-blue)](https://www.deeplearning.ai/short-courses/functions-tools-agents-langchain/)
[![LangChain](https://img.shields.io/badge/LangChain-Toolkit-yellow)](https://github.com/langchain-ai/langchain)
[![OpenAI](https://img.shields.io/badge/OpenAI-API-orange)](https://openai.com)

This repository contains code examples and exercises from the DeepLearning.AI short course **"Functions, Tools and Agents with LangChain"**, taught by **Harrison Chase**, Co-Founder and CEO of LangChain.

## 📚 Course Overview

The landscape of LLMs and the libraries that support them has evolved rapidly in recent months. This course is designed to keep you ahead of these changes. You'll explore new advancements like ChatGPT's function calling capability, and build a conversational agent using a new syntax called LangChain Expression Language (LCEL) for tasks like tagging, extraction, tool selection, and routing.

After taking this course, you'll know how to:
- Generate structured output, including function calls, using LLMs
- Use LCEL, which simplifies the customization of chains and agents, to build applications  
- Apply function calling to tasks like tagging and data extraction
- Understand tool selection and routing using LangChain tools and LLM function calling

## 🎯 What You'll Learn

- 🔧 **Learn about the most recent advancements in LLM APIs** - Stay current with the latest capabilities
- ⚡ **Use LangChain Expression Language (LCEL)** - A new syntax to compose and customize chains and agents faster
- 🤖 **Apply these new capabilities by building up a conversational agent** - Put theory into practice with hands-on projects

## 🗂️ Course Structure

This course is organized into **8 lessons**  with **6 code examples**, each accompanied by a Jupyter notebook:

### 1. **OpenAI Function Calling** – _Introduction to function calling with OpenAI's API_
- **Notebook:** [`L1-openai_functions_student.ipynb`](./L1/L1-openai_functions_student.ipynb)
- **Key Concepts:**
  - 🔍 **Basic Function Definitions**: Learn to define functions for LLM use
  - ⚙️ **Function Schema**: Structure function parameters and descriptions
  - 📝 **Example snippet:**
    ```python
    functions = [{
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
        }
    }]
    ```

### 2. **LangChain Expression Language (LCEL)** – _Composing chains with the new LCEL syntax_
- **Notebook:** [`L2-lcel-student.ipynb`](./L2/L2-lcel-student.ipynb)
- **Key Concepts & Code Examples:**
  - 🔗 **Chain Composition**: Using the pipe operator for cleaner syntax
  - 🗃️ **Runnable Maps**: Parallel processing and data flow
  - 📝 **Example snippet:**
    ```python
    chain = prompt | model | output_parser
    result = chain.invoke({"topic": "bears"})
    
    # More complex chain with retrieval
    chain = RunnableMap({
        "context": lambda x: retriever.get_relevant_documents(x["question"]),
        "question": lambda x: x["question"]
    }) | prompt | model | output_parser
    ```

### 3. **OpenAI Function Calling in LangChain** – _Integrating Pydantic models with LangChain_
- **Notebook:** [`L3-function-calling-student.ipynb`](./L3/L3-function-calling-student.ipynb)
- **Key Concepts & Code Examples:**
  - 🏗️ **Pydantic Integration**: Type-safe function definitions
  - 🔄 **Function Conversion**: Converting Pydantic models to OpenAI functions
  - 📝 **Example snippet:**
    ```python
    class WeatherSearch(BaseModel):
        """Call this with an airport code to get the weather at that airport"""
        airport_code: str = Field(description="airport code to get weather for")
    
    weather_function = convert_pydantic_to_openai_function(WeatherSearch)
    model_with_function = model.bind(functions=[weather_function])
    ```

### 4. **Tagging and Extraction** – _Using functions for structured data extraction_
- **Notebook:** [`L4-tagging-and-extraction-student.ipynb`](./L4/L4-tagging-and-extraction-student.ipynb)
- **Key Concepts & Code Examples:**
  - 🏷️ **Text Tagging**: Classify and tag text content
  - 📊 **Data Extraction**: Extract structured information from unstructured text
  - 📝 **Example snippet:**
    ```python
    class Tagging(BaseModel):
        """Tag the piece of text with particular info."""
        sentiment: str = Field(description="sentiment of text, should be `pos`, `neg`, or `neutral`")
        language: str = Field(description="language of text (should be ISO 639-1 code)")
    
    tagging_functions = [convert_pydantic_to_openai_function(Tagging)]
    ```
    ```python
    class Person(BaseModel):
    """Information about a person."""
        name: str = Field(description="person's name")
        age: Optional[int] = Field(description="person's age")
    
    class Information(BaseModel):
    """Information to extract."""
        people: List[Person] = Field(description="List of info about people")
    
    extraction_functions = [convert_pydantic_to_openai_function(Information)]
    ```

### 5. **Tools and Routing** – _Building tools and implementing routing logic_
- **Notebook:** [`L5-tools-routing-apis-student.ipynb`](./L5/L5-tools-routing-apis-student.ipynb)
- **Key Concepts & Code Examples:**
  - 🛠️ **Tool Creation**: Define custom tools with the `@tool` decorator
  - 🎯 **Routing Logic**: Direct queries to appropriate tools
  - 📝 **Example snippet:**
    ```python
    # Tool with custom input schema
    class SearchInput(BaseModel):
        query: str = Field(description="Thing to search for")
    
    @tool(args_schema=SearchInput)
    def search(query: str) -> str:
        """Search for the weather online."""
        return "42f"
    ```

### 6. **Conversational Agent** – _Building a complete conversational agent_
- **Notebook:** [`L6-functional_conversation-student.ipynb`](./L6/L6-functional_conversation-student.ipynb)
- **Key Concepts & Code Examples:**
  - 💬 **Agent Architecture**: Complete conversational agent implementation
  - 🌐 **Real API Integration**: Working with external APIs like OpenMeteo
  - 📝 **Example snippet:**
    ```python
    class OpenMeteoInput(BaseModel):
        latitude: float = Field(..., description="Latitude of the location to fetch weather data for")
        longitude: float = Field(..., description="Longitude of the location to fetch weather data for")
    
    @tool(args_schema=OpenMeteoInput)
    def get_current_temperature(latitude: float, longitude: float) -> dict:
        """Fetch current temperature for given coordinates."""
        # API integration code...
    ```

## 💻 Code Examples

A consolidated list of all notebooks and their purpose:

| Lesson | Notebook                                                                                    | Description                                 |
| ------ | ------------------------------------------------------------------------------------------- | ------------------------------------------- |
| 1      | [`L1-openai_functions_student.ipynb`](./L1/L1-openai_functions_student.ipynb)               | Introduction to OpenAI function calling     |
| 2      | [`L2-lcel-student.ipynb`](./L2/L2-lcel-student.ipynb)                                       | LangChain Expression Language (LCEL) syntax |
| 3      | [`L3-function-calling-student.ipynb`](./L3/L3-function-calling-student.ipynb)               | Function calling integration with LangChain |
| 4      | [`L4-tagging-and-extraction-student.ipynb`](./L4/L4-tagging-and-extraction-student.ipynb)   | Text tagging and data extraction techniques |
| 5      | [`L5-student.ipynb`](./L5/L5-student.ipynb)                                                 | Tools creation and routing implementation   |
| 6      | [`L6-functional_conversation-student.ipynb`](./L6/L6-functional_conversation-student.ipynb) | Complete conversational agent development   |

## 🚀 Getting Started

### Prerequisites

```bash
# Clone the repository
git clone https://github.com/duybaohuynhtan/Functions-Tools-and-Agents-with-LangChain.git
cd Functions-Tools-and-Agents-with-LangChain

# Install dependencies
pip install -r requirements.txt
```

### Environment Setup

You'll need an OpenAI API key. Create a `.env` file in the root directory:

   ```
   OPENAI_API_KEY=your_api_key_here
   ```

### Running the Examples
```bash
# Start Jupyter notebook
jupyter notebook
```

## 📋 Requirements

- **Python** ≥ 3.9.6
- **OpenAI** == 0.28.1
- **LangChain** == 0.0.312
- **Pydantic** == 1.10.8
- **Python-dotenv** == 1.0.0
- **tiktoken** == 0.5.1
- **faiss-cpu** == 1.7.4
- **Wikipedia** == 1.4.0
- **Panel** == 1.1.0

**Note**: This course uses specific versions for compatibility. The OpenAI library version (0.28.1) is intentionally older to match the course content.

## 🔗 Additional Resources

- 📖 [DeepLearning.AI Short Course Link](https://www.deeplearning.ai/short-courses/functions-tools-agents-langchain/)
- 🦜 [LangChain GitHub](https://github.com/langchain-ai/langchain)
- 📚 [LangChain Documentation](https://python.langchain.com/)
- 🤖 [OpenAI API Documentation](https://platform.openai.com/docs)

## 👨‍🏫 Instructor

- [**Harrison Chase**](https://www.linkedin.com/in/harrison-chase-961287118) - Co-Founder and CEO of LangChain

## 🙏 Acknowledgments

Special thanks to **DeepLearning.AI**, **Harrison Chase and the LangChain team** for creating such comprehensive learning materials.