import sys, re
sys.stdout.reconfigure(encoding='utf-8')
with open('E:/mivi on dataset/static/index.html', 'r', encoding='utf-8') as f:
    text = f.read()

askStream = '''
async function askStream(question) {
  const t = el("div", "typing", "Reading your question…");
  chat.appendChild(t); scroll();
  
  const m = el("div", "msg bot");
  chat.appendChild(m);
  
  try {
      const response = await fetch("/api/chat/stream", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({
              message: question,
              session_id: sessionId,
              history: chatHistory.slice(-6),
              profile: studentProfile
          })
      });
      
      if (response.status === 429) {
          throw new Error("Too many questions just now — give it a few seconds.");
      }
      if (response.status === 503) {
          throw new Error("The college data is still loading on the server. Try again in a moment.");
      }
      
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let finalText = "";
      
      while (true) {
          const {done, value} = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, {stream: true});
          const lines = buffer.split("\\n");
          buffer = lines.pop(); // keep incomplete line in buffer
          
          for (const line of lines) {
              if (line.startsWith("data: ")) {
                  try {
                      const data = JSON.parse(line.slice(6));
                      if (data.type === "token") {
                          finalText += data.content;
                          m.textContent = finalText;  // simple text update
                          scroll();
                      } else if (data.type === "done") {
                          if (data.session_id) sessionId = data.session_id;
                      } else if (data.type === "error") {
                          m.textContent = "Error: " + data.content;
                      }
                  } catch(e) {}
              }
          }
      }
      
      t.remove();
      chatHistory.push({role: "user", content: question},
                       {role: "assistant", content: finalText});
  } catch (err) {
      t.remove();
      m.textContent = "Error: " + err.message;
  }
}
'''

# insert askStream before ask()
idx = text.find('async function ask(')
text = text[:idx] + askStream + '\n' + text[idx:]

# update submit()
text = text.replace('await ask(text, "db");', 'await askStream(text);')

with open('E:/mivi on dataset/static/index.html', 'w', encoding='utf-8') as f:
    f.write(text)

print('Updated index.html')
