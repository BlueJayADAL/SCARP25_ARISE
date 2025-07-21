const apiKey = import.meta.env.VITE_XAI_API_KEY;
import axios from 'axios';

 const sendToGrokWithBackoff = async (text, retries = 3, baseDelay = 1000) => {
    for (let i = 0; i < retries; i++) {
      try {
        const response = await axios.post(
          'https://api.x.ai/v1/chat/completions',
          {
            model: 'grok',
            messages: [
              { role: 'system', content: 'You are Grok, a helpful AI assistant.' },
              { role: 'user', content: text },
            ],
            max_tokens: 50, // Reduced to minimize token usage
          },
          {
            headers: {
              'Content-Type': 'application/json',
              Authorization: `Bearer ${import.meta.env.VITE_XAI_API_KEY}`,
            },
          }
        );
        return response.data.choices[0].message.content.trim();
      } catch (error) {
        if (error.response?.status === 429) {
          const retryAfter = error.response.headers['retry-after']
            ? parseInt(error.response.headers['retry-after']) * 1000
            : baseDelay * Math.pow(2, i); // Exponential backoff
          console.warn(`Rate limit hit, retrying after ${retryAfter}ms...`);
          await sleep(retryAfter);
          continue;
        }
        console.error('Error communicating with Grok API:', error);
        setError('Failed to get response from Grok. Please try again later.');
        return 'Sorry, something went wrong.';
      }
    }
    setError('Max retries reached. Please try again later.');
    return 'Sorry, something went wrong.';
  };

  export default sendToGrokWithBackoff;