# Example: Conversational AI Usage

from src.telecom_ai_platform.agents.conversational_ai import ConversationalAI
import asyncio

async def chat_example():
    """Example of conversational AI interaction"""
    
    # Initialize the AI agent
    agent = ConversationalAI()
    
    # Example conversation flow
    queries = [
        "Show me RSRP anomalies for site 001 in the last week",
        "What's the trend for downlink throughput?",
        "Compare this with site 002",
        "Generate a summary report",
        "Are there any critical issues I should know about?"
    ]
    
    conversation_id = None
    
    for query in queries:
        print(f"\nUser: {query}")
        
        # Process the query
        response = await agent.process_query(
            query, 
            conversation_id=conversation_id
        )
        
        conversation_id = response.get('conversation_id')
        
        print(f"AI: {response['message']}")
        
        # Show tool calls if any
        if response.get('tool_calls'):
            print("Tools used:")
            for tool_call in response['tool_calls']:
                print(f"  - {tool_call['tool']}: {tool_call['parameters']}")
        
        # Show follow-up suggestions
        if response.get('follow_up_suggestions'):
            print("Suggested follow-ups:")
            for suggestion in response['follow_up_suggestions']:
                print(f"  - {suggestion}")
        
        print("-" * 50)

def simple_chat():
    """Simple synchronous chat example"""
    agent = ConversationalAI()
    
    # Single query processing
    query = "Find performance issues in the northern sector"
    response = agent.process_simple_query(query)
    
    print(f"Query: {query}")
    print(f"Response: {response['message']}")
    
    if response.get('data'):
        print(f"Data points found: {len(response['data'])}")

if __name__ == "__main__":
    print("Running conversational AI examples...")
    
    # Run simple example
    simple_chat()
    
    # Run async conversation example
    asyncio.run(chat_example())
