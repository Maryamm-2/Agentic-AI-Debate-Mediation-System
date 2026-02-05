from agents.heat import HeatDetector

samples = [
	"I completely disagree with you! Your argument is absurd and dangerous!!!",
	"I understand your point, but perhaps we should look at the data more closely.",
	"THIS IS AN OUTRAGE and everyone should be alarmed!!!",
	"I'm concerned about the potential harms, but let's discuss civilly.",
	"You're an idiot and your proposal is garbage!"
]

hd = HeatDetector()
for s in samples:
	metrics = hd.analyze_heat(s, "Tester")
	print(f"Message: {s}\n Heat: {metrics.heat_score:.3f}, Emotion: {metrics.emotion_level.value}, Primary: {metrics.primary_emotion}, Intensity: {metrics.emotional_intensity:.3f}, Indicators: {metrics.aggression_indicators}\n")
