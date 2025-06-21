# Personal Language Model - Browser AI

🤖 **Train and chat with your own AI language model - runs entirely in your browser!**

[![GitHub Pages](https://img.shields.io/badge/demo-live-brightgreen)](https://yourusername.github.io/PersonalLanguageModel/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Features

- 🌐 **Runs entirely in browser** - No server setup required
- 🔒 **Complete privacy** - Your data never leaves your device
- ⚡ **No installation** - Just open the webpage and start using
- 🎓 **Custom training** - Train the AI on your own text data
- 💬 **Interactive chat** - Have conversations with your trained model
- 📱 **Mobile friendly** - Works on phones, tablets, and desktops

## 🚀 Quick Start

### Option 1: Use the Live Demo
Visit the live demo: **[https://yourusername.github.io/PersonalLanguageModel/](https://yourusername.github.io/PersonalLanguageModel/)**

### Option 2: Host on GitHub Pages (Recommended)

1. **Fork this repository** to your GitHub account
2. **Enable GitHub Pages**:
   - Go to your repository settings
   - Navigate to "Pages" section
   - Set source to "Deploy from a branch"
   - Select "main" branch and "/ (root)" folder
   - Save settings
3. **Access your site** at `https://yourusername.github.io/PersonalLanguageModel/`

That's it! No backend setup, no API keys, no complex deployment.

### Option 3: Run Locally

1. Clone the repository:
```bash
git clone https://github.com/yourusername/PersonalLanguageModel.git
cd PersonalLanguageModel
```

2. Serve the files (any method works):
```bash
# Python
python -m http.server 8000

# Node.js
npx serve .

# Or just open index.html in your browser
```

## 🎯 How to Use

### 1. Chat with the Pre-trained Model
- Open the webpage
- Wait for the AI to load (30-60 seconds on first visit)
- Start chatting in the "Chat" tab
- Try the quick action buttons for inspiration

### 2. Train Your Own Model
- Switch to the "Train Model" tab
- Enter your training text (minimum 200 characters)
- Choose model size and training parameters
- Click "Start Training" and wait for completion
- Switch back to chat and talk with your custom AI!

### 3. Training Tips
- **More text = better results** (aim for 1000+ characters)
- **Consistent style** helps the model learn patterns
- **Medium model size** offers the best balance
- **3-5 epochs** are usually sufficient

## 🛠️ Technical Details

### Architecture
- **Frontend**: Pure HTML/CSS/JavaScript
- **AI Engine**: PyTorch running via Pyodide (WebAssembly)
- **Model**: Character-level Transformer
- **Deployment**: Static hosting (GitHub Pages, Netlify, etc.)

### Browser Requirements
- Modern browser with WebAssembly support
- 2GB+ RAM recommended
- Internet connection for initial library download

### Model Specifications
- **Small**: 32-dim embeddings, 2 heads, 1 layer (~50K parameters)
- **Medium**: 64-dim embeddings, 4 heads, 2 layers (~200K parameters)  
- **Large**: 128-dim embeddings, 8 heads, 3 layers (~800K parameters)

## 📁 Project Structure

```
PersonalLanguageModel/
├── index.html              # Main application (browser AI)
├── README.md               # This file
├── api.py                  # Optional: Server API (for advanced users)
├── quick_chat.html         # Simple chat interface
├── unified-chat.html       # Dual browser/server interface
└── docs/                   # Additional documentation
```

## 🔧 Customization

### Modify the Default Training Text
Edit the `demoText` variable in `index.html`:
```javascript
const demoText = `Your custom training text here...`;
```

### Adjust Model Parameters
Modify the `configs` object in the training function:
```javascript
configs = {
    "custom": {"embed_dim": 96, "num_heads": 6, "num_layers": 2, "sequence_length": 35}
}
```

### Change the UI Theme
Update the CSS variables in the `<style>` section:
```css
:root {
    --primary-color: #your-color;
    --background-gradient: linear-gradient(135deg, #color1, #color2);
}
```

## 🚀 Deployment Options

### GitHub Pages (Easiest)
1. Fork repository
2. Enable Pages in settings
3. Done! ✅

### Netlify
1. Connect your GitHub repository
2. Deploy automatically
3. Get custom domain

### Vercel
1. Import GitHub repository
2. Deploy with one click
3. Automatic HTTPS

### Any Static Host
Upload the files to any web server that serves static files.

## 🤝 Contributing

Contributions are welcome! Here are some ideas:

- 🎨 UI/UX improvements
- 🧠 Better model architectures
- 📚 More training examples
- 🌍 Internationalization
- 📱 Mobile app wrapper
- 🔧 Performance optimizations

### Development Setup
1. Fork and clone the repository
2. Make your changes
3. Test locally by serving the files
4. Submit a pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Pyodide](https://pyodide.org/) - Python in the browser
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Transformer architecture](https://arxiv.org/abs/1706.03762) - Attention is all you need

## 📞 Support

- 🐛 **Bug reports**: Open an issue on GitHub
- 💡 **Feature requests**: Start a discussion
- ❓ **Questions**: Check the FAQ or open an issue
- 💬 **Community**: Join our discussions

---

**Made with ❤️ for the AI community**

*Train your own AI in minutes, not hours!*