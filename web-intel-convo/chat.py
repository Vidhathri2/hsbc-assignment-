import uuid
import cmd
from agents.rag_agent import ConversationalRAGAgent
from colorama import init, Fore, Style

init(autoreset=True)

class ChatCLI(cmd.Cmd):
    intro = f"{Fore.CYAN}{Style.BRIGHT}Welcome to the Web Intelligence RAG Chat!{Style.RESET_ALL}\nType 'help' or '?' to list commands. Type 'exit' or 'quit' to end."
    prompt = f"{Fore.GREEN}You: {Style.RESET_ALL}"

    def __init__(self):
        super().__init__()
        self.rag_agent = ConversationalRAGAgent()
        self.session_id = str(uuid.uuid4())
        print(f"\nStarted new session: {self.session_id}\n")

    def default(self, line):
        if line.lower() in ['exit', 'quit']:
            return self.do_exit(line)
            
        print(f"{Fore.YELLOW}Thinking...{Style.RESET_ALL}")
        answer, sources = self.rag_agent.chat(self.session_id, line)
        
        print(f"\n{Fore.CYAN}Agent:{Style.RESET_ALL} {answer}")
        
        if sources:
            print(f"\n{Fore.MAGENTA}Sources:{Style.RESET_ALL}")
            for i, source in enumerate(sources):
                url = source['metadata'].get('url', 'Unknown URL')
                print(f"  [{i+1}] {url}")
        print("\n")

    def do_new_session(self, arg):
        """Starts a new chat session, clearing the history."""
        self.session_id = str(uuid.uuid4())
        print(f"\n{Fore.GREEN}Started new session: {self.session_id}{Style.RESET_ALL}\n")

    def do_exit(self, arg):
        """Exits the chat interface."""
        print(f"{Fore.CYAN}Goodbye!{Style.RESET_ALL}")
        return True
        
    def do_quit(self, arg):
        """Exits the chat interface."""
        return self.do_exit(arg)

if __name__ == "__main__":
    ChatCLI().cmdloop()
