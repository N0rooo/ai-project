"use client"

import { useRef, useState } from "react"

export default function Home() {
	const [prompt, setPrompt] = useState("")
	const [generatedText, setGeneratedText] = useState("")
	const [loading, setLoading] = useState(false)
	const [speaking, setSpeaking] = useState(false)
	const [error, setError] = useState<string | null>(null)
	const audioRef = useRef<HTMLAudioElement | null>(null)
	const speechSynthesis = typeof window !== "undefined" ? window.speechSynthesis : null

	const stopSpeaking = () => {
		if (speechSynthesis) {
			speechSynthesis.cancel();
			setSpeaking(false);
		}
	}

	const speakWithWebSpeech = (text: string) => {
		if (!speechSynthesis || !text.trim()) return;
		
		speechSynthesis.cancel();
		
		const sentences = text.match(/[^.!?]+[.!?]+/g) || [text];
		let currentIndex = 0;

		const speakNextSentence = () => {
			if (currentIndex >= sentences.length) {
				setSpeaking(false);
				return;
			}

			const utterance = new SpeechSynthesisUtterance(sentences[currentIndex].trim());
			utterance.lang = 'en-GB';
			utterance.rate = 0.9;
			utterance.pitch = 1.0;
			utterance.volume = 1.0;

			utterance.onstart = () => setSpeaking(true);
			utterance.onend = () => {
				currentIndex++;
				speakNextSentence();
			};
			utterance.onerror = () => {
				setSpeaking(false);
				setError('Speech synthesis failed');
			};

			speechSynthesis.speak(utterance);
		};

		speakNextSentence();
	}

	const generateText = async () => {
		try {
			setLoading(true)
			const response = await fetch("http://localhost:8000/generate", {
				method: "POST",
				headers: {
					"Content-Type": "application/json",
				},
				body: JSON.stringify({
					prompt,
					max_length: 100,
					temperature: 1,
				}),
			})
			const data = await response.json()
			setGeneratedText(data.generated_text)
		} catch (error) {
			console.error("Error:", error)
			setError("Failed to generate text. Please try again.")
		} finally {
			setLoading(false)
		}
	}

	return (
		<div className="min-h-screen p-8 bg-gray-900 text-white">
			<main className="max-w-2xl mx-auto space-y-8">
				<div className="text-center">
					<h1 className="text-4xl font-bold mb-4">Movie Dialog Generator</h1>
					<p className="text-gray-400">Enter a prompt to generate movie-style dialogue</p>
				</div>

				<div className="space-y-4">
					<input 
						type="text" 
						value={prompt} 
						onChange={(e) => setPrompt(e.target.value)} 
						placeholder="Enter your prompt" 
						className="w-full p-3 border rounded bg-gray-800 border-gray-700 text-white placeholder-gray-500"
					/>
					<button 
						onClick={generateText} 
						disabled={loading} 
						className="w-full bg-red-800 text-white p-3 rounded hover:bg-red-700 disabled:bg-gray-700 transition-colors"
					>
						{loading ? "Generating..." : "Generate Text"}
					</button>
				</div>

				{generatedText && (
					<div className="bg-gray-800 p-6 rounded-lg border border-gray-700">
						<div className="flex justify-between items-center mb-4">
							<h2 className="font-bold text-xl">Generated Text:</h2>
							{speaking ? (
								<button
									onClick={stopSpeaking}
									className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 transition-colors"
								>
									Stop Speaking
								</button>
							) : (
								<button
									onClick={() => speakWithWebSpeech(generatedText)}
									className="px-4 py-2 bg-gray-700 text-white rounded hover:bg-gray-600 transition-colors"
								>
									Speak Text
								</button>
							)}
						</div>
						<p className="italic text-gray-300">{generatedText}</p>
					</div>
				)}
				
				{error && (
					<div className="text-red-500 text-center bg-gray-800 p-4 rounded-lg">
						{error}
					</div>
				)}

			
			</main>
		</div>
	)
}
