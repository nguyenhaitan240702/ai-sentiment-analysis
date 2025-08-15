"""
Conversation Sentiment Analysis Service
Advanced conversation-level sentiment analysis with context awareness
"""

import logging
import time
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import statistics
import re

from apps.api.schemas.request import (
    ConversationSentimentRequest,
    ConversationMessage,
    ConversationContext
)
from apps.api.schemas.response import (
    ConversationSentimentResponse,
    SentimentResult,
    EmotionalFlow,
    SentimentTurningPoint,
    ParticipantSentiment,
    EscalationAnalysis,
    ConversationSummary,
    ConversationQuality,
    MessageAnalysis
)
from apps.api.services.sentiment_service import SentimentService
from core.models.base import ModelManager

logger = logging.getLogger(__name__)

class ConversationSentimentService:
    """Advanced conversation sentiment analysis service"""

    def __init__(self):
        self.sentiment_service: Optional[SentimentService] = None

        # Emotion keywords for detection
        self.emotion_keywords = {
            'anger': ['giận', 'tức', 'bực', 'phẫn nộ', 'cáu', 'gắt', 'nóng', 'điên'],
            'frustration': ['thất vọng', 'bực mình', 'khó chịu', 'phiền', 'tức tối'],
            'satisfaction': ['hài lòng', 'vui', 'thích', 'tuyệt', 'tốt', 'ổn'],
            'empathy': ['hiểu', 'thông cảm', 'chia sẻ', 'đồng cảm', 'cảm thông'],
            'gratitude': ['cảm ơn', 'thanks', 'appreciate', 'grateful'],
            'concern': ['lo lắng', 'quan tâm', 'băn khoăn', 'worry'],
            'relief': ['nhẹ nhõm', 'yên tâm', 'thở phào', 'an tâm']
        }

        # Intent patterns
        self.intent_patterns = {
            'complaint': ['khiếu nại', 'phàn nán', 'không hài lòng', 'vấn đề', 'lỗi'],
            'request': ['yêu cầu', 'cần', 'muốn', 'mong', 'xin'],
            'question': ['?', 'sao', 'thế nào', 'như thế', 'why', 'how'],
            'acknowledgment': ['hiểu rồi', 'được', 'ok', 'cảm ơn', 'tôi sẽ'],
            'solution': ['giải pháp', 'cách', 'method', 'giải quyết', 'fix'],
            'apology': ['xin lỗi', 'sorry', 'apologize', 'thành thật']
        }

        # Domain-specific weights
        self.domain_weights = {
            'customer_service': {'resolution_focus': 1.2, 'satisfaction_weight': 1.5},
            'sales': {'enthusiasm_boost': 1.1, 'objection_weight': 1.3},
            'support': {'technical_weight': 1.0, 'patience_weight': 1.2},
            'general': {'balanced_weight': 1.0}
        }

    async def initialize(self, sentiment_service: SentimentService):
        """Initialize with sentiment service"""
        self.sentiment_service = sentiment_service
        logger.info("Conversation sentiment service initialized")

    async def analyze_conversation(
        self,
        request: ConversationSentimentRequest
    ) -> ConversationSentimentResponse:
        """Main conversation analysis method"""
        if not self.sentiment_service:
            raise RuntimeError("Service not initialized")

        start_time = time.time()

        # Set default analysis options
        options = request.analysis_options or {}
        context = request.context or ConversationContext()

        # Step 1: Analyze individual messages
        message_analyses = await self._analyze_individual_messages(
            request.conversation, context
        )

        # Step 2: Emotional flow analysis
        emotional_flow = self._analyze_emotional_flow(message_analyses)

        # Step 3: Participant analysis
        participants = self._analyze_participants(message_analyses, context)

        # Step 4: Escalation analysis
        escalation_analysis = self._analyze_escalation(message_analyses)

        # Step 5: Content summary
        conversation_summary = self._analyze_content_summary(
            message_analyses, context
        )

        # Step 6: Quality assessment
        conversation_quality = self._assess_conversation_quality(
            message_analyses, participants, escalation_analysis
        )

        # Step 7: Overall sentiment
        overall_sentiment = self._calculate_overall_sentiment(
            message_analyses, emotional_flow, context
        )

        # Step 8: Conversation type detection
        conversation_type = self._detect_conversation_type(
            message_analyses, context, escalation_analysis
        )

        # Metadata
        analysis_metadata = {
            'processing_time_ms': int((time.time() - start_time) * 1000),
            'message_count': len(request.conversation),
            'participant_count': len(set(msg.speaker for msg in request.conversation)),
            'analysis_version': '2.0',
            'features_used': self._get_features_used(options)
        }

        return ConversationSentimentResponse(
            overall_sentiment=overall_sentiment,
            conversation_type=conversation_type,
            emotional_flow=emotional_flow,
            participants=participants,
            escalation_analysis=escalation_analysis,
            conversation_summary=conversation_summary,
            conversation_quality=conversation_quality,
            message_analysis=message_analyses,
            analysis_metadata=analysis_metadata
        )

    async def _analyze_individual_messages(
        self,
        messages: List[ConversationMessage],
        context: ConversationContext
    ) -> List[MessageAnalysis]:
        """Analyze each message individually with context"""
        analyses = []

        for i, message in enumerate(messages):
            # Get basic sentiment
            sentiment_result = await self.sentiment_service.analyze_text(
                message.text, context.language
            )

            # Convert to SentimentResult format
            sentiment = SentimentResult(
                label=sentiment_result.label,
                score=sentiment_result.score,
                scores=sentiment_result.scores
            )

            # Detect emotions
            emotions = self._detect_emotions(message.text)

            # Detect intent
            intent = self._detect_intent(message.text)

            # Context factors
            context_factors = self._analyze_context_factors(
                message, i, messages, context
            )

            # Response tracking
            response_to = self._find_response_to(message, i, messages)

            analysis = MessageAnalysis(
                message_index=i,
                speaker=message.speaker,
                text=message.text,
                sentiment=sentiment,
                emotions=emotions,
                intent=intent,
                context_factors=context_factors,
                response_to=response_to
            )

            analyses.append(analysis)

        return analyses

    def _analyze_emotional_flow(
        self,
        message_analyses: List[MessageAnalysis]
    ) -> EmotionalFlow:
        """Analyze emotional progression throughout conversation"""

        # Extract sentiment scores
        scores = [self._sentiment_to_score(msg.sentiment) for msg in message_analyses]

        # Calculate trajectory
        trajectory = self._calculate_trajectory(scores)

        # Assess stability
        stability = self._assess_stability(scores)

        # Find dominant emotion
        all_emotions = []
        for msg in message_analyses:
            all_emotions.extend(msg.emotions)
        dominant_emotion = max(set(all_emotions), key=all_emotions.count) if all_emotions else "neutral"

        # Detect turning points
        turning_points = self._detect_turning_points(message_analyses)

        return EmotionalFlow(
            trajectory=trajectory,
            stability=stability,
            dominant_emotion=dominant_emotion,
            turning_points=turning_points,
            sentiment_progression=scores
        )

    def _analyze_participants(
        self,
        message_analyses: List[MessageAnalysis],
        context: ConversationContext
    ) -> Dict[str, ParticipantSentiment]:
        """Analyze sentiment for each participant"""
        participants = {}

        # Group messages by speaker
        speaker_messages = {}
        for msg in message_analyses:
            if msg.speaker not in speaker_messages:
                speaker_messages[msg.speaker] = []
            speaker_messages[msg.speaker].append(msg)

        # Analyze each participant
        for speaker, messages in speaker_messages.items():
            # Calculate overall sentiment
            scores = [self._sentiment_to_score(msg.sentiment) for msg in messages]
            overall_score = sum(scores) / len(scores)
            overall_sentiment = self._score_to_label(overall_score)

            # Extract emotions
            all_emotions = []
            for msg in messages:
                all_emotions.extend(msg.emotions)
            dominant_emotions = list(set(all_emotions))[:3]  # Top 3 unique emotions

            # Analyze attitude and communication style
            attitude = self._analyze_attitude(messages, context)
            communication_style = self._analyze_communication_style(messages)
            engagement_level = self._analyze_engagement_level(messages)

            participants[speaker] = ParticipantSentiment(
                overall_sentiment=overall_sentiment,
                overall_score=overall_score,
                dominant_emotions=dominant_emotions,
                sentiment_progression=scores,
                attitude=attitude,
                communication_style=communication_style,
                engagement_level=engagement_level
            )

        return participants

    def _analyze_escalation(
        self,
        message_analyses: List[MessageAnalysis]
    ) -> EscalationAnalysis:
        """Analyze escalation patterns"""

        # Calculate escalation indicators
        scores = [self._sentiment_to_score(msg.sentiment) for msg in message_analyses]

        # Find peak tension point
        peak_tension_point = None
        if scores:
            min_score_idx = scores.index(min(scores))
            if scores[min_score_idx] < -0.3:
                peak_tension_point = min_score_idx

        # Analyze trend
        escalation_trend = self._analyze_escalation_trend(scores)

        # Current escalation level
        current_score = scores[-1] if scores else 0
        escalation_level = self._score_to_escalation_level(current_score)

        # Find resolution indicators
        resolution_indicators = self._find_resolution_indicators(message_analyses)

        # Calculate conflict intensity
        conflict_intensity = self._calculate_conflict_intensity(scores, message_analyses)

        return EscalationAnalysis(
            escalation_level=escalation_level,
            escalation_trend=escalation_trend,
            peak_tension_point=peak_tension_point,
            resolution_indicators=resolution_indicators,
            conflict_intensity=conflict_intensity
        )

    def _analyze_content_summary(
        self,
        message_analyses: List[MessageAnalysis],
        context: ConversationContext
    ) -> ConversationSummary:
        """Summarize conversation content"""

        # Extract main topics (simplified keyword extraction)
        all_text = " ".join([msg.text for msg in message_analyses])
        main_topics = self._extract_topics(all_text, context)

        # Identify key issues
        key_issues = self._identify_issues(message_analyses)

        # Determine resolution status
        resolution_status = self._determine_resolution_status(message_analyses)

        # Overall outcome
        final_sentiments = message_analyses[-3:] if len(message_analyses) >= 3 else message_analyses
        avg_final_score = sum(self._sentiment_to_score(msg.sentiment) for msg in final_sentiments) / len(final_sentiments)
        conversation_outcome = self._score_to_label(avg_final_score)

        # Satisfaction indicators
        satisfaction_indicators = self._find_satisfaction_indicators(message_analyses)

        return ConversationSummary(
            main_topics=main_topics,
            key_issues=key_issues,
            resolution_status=resolution_status,
            conversation_outcome=conversation_outcome,
            satisfaction_indicators=satisfaction_indicators
        )

    def _assess_conversation_quality(
        self,
        message_analyses: List[MessageAnalysis],
        participants: Dict[str, ParticipantSentiment],
        escalation_analysis: EscalationAnalysis
    ) -> ConversationQuality:
        """Assess overall conversation quality"""

        # Communication quality based on sentiment progression
        avg_sentiment = sum(self._sentiment_to_score(msg.sentiment) for msg in message_analyses) / len(message_analyses)
        communication_quality = self._score_to_quality_level(avg_sentiment)

        # Professionalism level
        professionalism_level = self._assess_professionalism(message_analyses)

        # Empathy score
        empathy_score = self._calculate_empathy_score(message_analyses)

        # Solution effectiveness
        solution_effectiveness = self._assess_solution_effectiveness(message_analyses)

        # Customer satisfaction score
        customer_satisfaction_score = self._estimate_customer_satisfaction(
            participants, escalation_analysis
        )

        return ConversationQuality(
            communication_quality=communication_quality,
            professionalism_level=professionalism_level,
            empathy_score=empathy_score,
            solution_effectiveness=solution_effectiveness,
            customer_satisfaction_score=customer_satisfaction_score
        )

    def _calculate_overall_sentiment(
        self,
        message_analyses: List[MessageAnalysis],
        emotional_flow: EmotionalFlow,
        context: ConversationContext
    ) -> SentimentResult:
        """Calculate weighted overall conversation sentiment"""

        scores = [self._sentiment_to_score(msg.sentiment) for msg in message_analyses]

        # Apply temporal weighting (recent messages more important)
        weighted_scores = []
        for i, score in enumerate(scores):
            temporal_weight = 0.5 + 0.5 * (i / len(scores))  # 0.5 to 1.0
            weighted_scores.append(score * temporal_weight)

        # Apply domain-specific weighting
        domain = context.domain or 'general'
        domain_modifier = self.domain_weights.get(domain, {'balanced_weight': 1.0})

        # Calculate weighted average
        overall_score = sum(weighted_scores) / len(weighted_scores)

        # Adjust based on emotional flow stability
        if emotional_flow.stability == "improving":
            overall_score += 0.1
        elif emotional_flow.stability == "deteriorating":
            overall_score -= 0.1

        # Convert to label and confidence
        label = self._score_to_label(overall_score)
        confidence = min(abs(overall_score) + 0.5, 1.0)

        # Calculate detailed scores
        scores_dict = self._calculate_detailed_scores(overall_score, confidence)

        return SentimentResult(
            label=label,
            score=confidence,
            scores=scores_dict
        )

    # Helper methods
    def _sentiment_to_score(self, sentiment: SentimentResult) -> float:
        """Convert sentiment to numeric score"""
        if sentiment.label == "positive":
            return sentiment.score
        elif sentiment.label == "negative":
            return -sentiment.score
        else:
            return 0.0

    def _score_to_label(self, score: float) -> str:
        """Convert numeric score to sentiment label"""
        if score > 0.15:
            return "positive"
        elif score < -0.15:
            return "negative"
        else:
            return "neutral"

    def _detect_emotions(self, text: str) -> List[str]:
        """Detect emotions in text"""
        text_lower = text.lower()
        detected = []

        for emotion, keywords in self.emotion_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                detected.append(emotion)

        return detected

    def _detect_intent(self, text: str) -> Optional[str]:
        """Detect message intent"""
        text_lower = text.lower()

        for intent, patterns in self.intent_patterns.items():
            if any(pattern in text_lower for pattern in patterns):
                return intent

        return None

    def _analyze_context_factors(
        self,
        message: ConversationMessage,
        index: int,
        all_messages: List[ConversationMessage],
        context: ConversationContext
    ) -> List[str]:
        """Analyze contextual factors affecting sentiment"""
        factors = []

        # Position in conversation
        if index == 0:
            factors.append("opening_message")
        elif index == len(all_messages) - 1:
            factors.append("closing_message")

        # Message length
        if len(message.text) > 100:
            factors.append("detailed_message")
        elif len(message.text) < 20:
            factors.append("brief_message")

        # Urgency indicators
        if any(word in message.text.lower() for word in ['urgent', 'khẩn cấp', 'ngay', 'immediately']):
            factors.append("urgency")

        # Time-based factors
        if message.timestamp:
            hour = message.timestamp.hour
            if 9 <= hour <= 17:
                factors.append("business_hours")
            else:
                factors.append("after_hours")

        return factors

    def _find_response_to(
        self,
        message: ConversationMessage,
        index: int,
        all_messages: List[ConversationMessage]
    ) -> Optional[int]:
        """Find which message this is responding to"""
        if index == 0:
            return None

        # Simple heuristic: responds to the most recent message from different speaker
        for i in range(index - 1, -1, -1):
            if all_messages[i].speaker != message.speaker:
                return i

        return None

    def _calculate_trajectory(self, scores: List[float]) -> str:
        """Calculate emotional trajectory"""
        if len(scores) < 2:
            return "stable"

        # Analyze trend
        start_avg = sum(scores[:len(scores)//3]) / max(len(scores)//3, 1)
        end_avg = sum(scores[-len(scores)//3:]) / max(len(scores)//3, 1)

        diff = end_avg - start_avg

        if diff > 0.3:
            return "improving"
        elif diff < -0.3:
            return "deteriorating"
        else:
            return "stable"

    def _assess_stability(self, scores: List[float]) -> str:
        """Assess emotional stability"""
        if len(scores) < 2:
            return "stable"

        variance = statistics.variance(scores)

        if variance > 0.5:
            return "volatile"
        elif variance > 0.2:
            return "unstable"
        else:
            return "stable"

    def _detect_turning_points(
        self,
        message_analyses: List[MessageAnalysis]
    ) -> List[SentimentTurningPoint]:
        """Detect significant sentiment changes"""
        turning_points = []

        if len(message_analyses) < 2:
            return turning_points

        for i in range(1, len(message_analyses)):
            prev_sentiment = message_analyses[i-1].sentiment.label
            curr_sentiment = message_analyses[i].sentiment.label

            if prev_sentiment != curr_sentiment and curr_sentiment != "neutral":
                # Determine trigger
                trigger = self._determine_trigger(message_analyses[i])

                turning_point = SentimentTurningPoint(
                    message_index=i,
                    from_sentiment=prev_sentiment,
                    to_sentiment=curr_sentiment,
                    trigger=trigger,
                    confidence=0.8
                )
                turning_points.append(turning_point)

        return turning_points

    def _determine_trigger(self, message_analysis: MessageAnalysis) -> str:
        """Determine what triggered sentiment change"""
        if message_analysis.intent:
            return message_analysis.intent
        elif "solution" in message_analysis.text.lower():
            return "solution_provided"
        elif "xin lỗi" in message_analysis.text.lower() or "sorry" in message_analysis.text.lower():
            return "apology"
        elif "hiểu" in message_analysis.text.lower():
            return "acknowledgment"
        else:
            return "unknown"

    def _detect_conversation_type(
        self,
        message_analyses: List[MessageAnalysis],
        context: ConversationContext,
        escalation_analysis: EscalationAnalysis
    ) -> str:
        """Detect type of conversation"""

        # Use context if available
        if context.conversation_type:
            return context.conversation_type

        # Analyze patterns
        intents = [msg.intent for msg in message_analyses if msg.intent]

        if "complaint" in intents and escalation_analysis.escalation_level in ["resolved", "none"]:
            return "customer_service_resolution"
        elif "complaint" in intents:
            return "customer_complaint"
        elif "question" in intents and "solution" in intents:
            return "support_inquiry"
        elif context.domain == "sales":
            return "sales_interaction"
        else:
            return "general_conversation"

    def _get_features_used(self, options: Dict) -> List[str]:
        """Get list of analysis features used"""
        features = ["basic_sentiment", "emotional_flow", "participant_analysis"]

        # Handle both dict and Pydantic model
        if hasattr(options, 'include_escalation_tracking'):
            # Pydantic model
            if getattr(options, 'include_escalation_tracking', True):
                features.append("escalation_tracking")
            if getattr(options, 'include_summary', True):
                features.append("content_summary")
        else:
            # Dictionary
            if options.get("include_escalation_tracking", True):
                features.append("escalation_tracking")
            if options.get("include_summary", True):
                features.append("content_summary")

        return features

    # Additional helper methods for quality assessment
    def _score_to_escalation_level(self, score: float) -> str:
        """Convert score to escalation level"""
        if score < -0.7:
            return "critical"
        elif score < -0.4:
            return "high"
        elif score < -0.1:
            return "medium"
        elif score < 0.1:
            return "low"
        else:
            return "none"

    def _analyze_escalation_trend(self, scores: List[float]) -> str:
        """Analyze escalation trend"""
        if len(scores) < 2:
            return "stable"

        recent_trend = scores[-3:] if len(scores) >= 3 else scores
        if len(recent_trend) >= 2:
            if recent_trend[-1] > recent_trend[0] + 0.2:
                return "de-escalating"
            elif recent_trend[-1] < recent_trend[0] - 0.2:
                return "escalating"

        return "stable"

    def _find_resolution_indicators(self, message_analyses: List[MessageAnalysis]) -> List[str]:
        """Find indicators of issue resolution"""
        indicators = []

        for msg in message_analyses:
            text_lower = msg.text.lower()
            if any(word in text_lower for word in ['giải quyết', 'resolved', 'fixed', 'solution']):
                indicators.append("solution_provided")
            if any(word in text_lower for word in ['cảm ơn', 'thank', 'appreciate']):
                indicators.append("gratitude_expressed")
            if any(word in text_lower for word in ['hài lòng', 'satisfied', 'happy']):
                indicators.append("satisfaction_expressed")

        return list(set(indicators))

    def _calculate_conflict_intensity(
        self,
        scores: List[float],
        message_analyses: List[MessageAnalysis]
    ) -> float:
        """Calculate overall conflict intensity"""
        if not scores:
            return 0.0

        # Base intensity from sentiment variance
        base_intensity = min(statistics.variance(scores) if len(scores) > 1 else 0, 1.0)

        # Boost for negative emotions
        negative_emotions = ['anger', 'frustration']
        emotion_boost = 0
        for msg in message_analyses:
            if any(emotion in msg.emotions for emotion in negative_emotions):
                emotion_boost += 0.1

        return min(base_intensity + emotion_boost, 1.0)

    def _extract_topics(self, text: str, context: ConversationContext) -> List[str]:
        """Extract main topics (simplified)"""
        topics = []

        # Domain-specific topic detection
        if context.domain == "customer_service":
            if any(word in text.lower() for word in ['sản phẩm', 'product']):
                topics.append("product_issue")
            if any(word in text.lower() for word in ['dịch vụ', 'service']):
                topics.append("service_quality")
            if any(word in text.lower() for word in ['thanh toán', 'payment', 'billing']):
                topics.append("billing")

        return topics or ["general"]

    def _identify_issues(self, message_analyses: List[MessageAnalysis]) -> List[str]:
        """Identify key issues raised"""
        issues = []

        for msg in message_analyses:
            if msg.intent == "complaint":
                text_lower = msg.text.lower()
                if any(word in text_lower for word in ['lỗi', 'error', 'bug']):
                    issues.append("technical_issue")
                if any(word in text_lower for word in ['chậm', 'slow', 'delay']):
                    issues.append("performance_issue")
                if any(word in text_lower for word in ['không hoạt động', 'not working']):
                    issues.append("functionality_issue")

        return list(set(issues))

    def _determine_resolution_status(self, message_analyses: List[MessageAnalysis]) -> str:
        """Determine if issues were resolved"""
        resolution_indicators = self._find_resolution_indicators(message_analyses)

        if "solution_provided" in resolution_indicators and "satisfaction_expressed" in resolution_indicators:
            return "resolved"
        elif "solution_provided" in resolution_indicators:
            return "partially_resolved"
        else:
            return "unresolved"

    def _find_satisfaction_indicators(self, message_analyses: List[MessageAnalysis]) -> List[str]:
        """Find satisfaction/dissatisfaction indicators"""
        indicators = []

        for msg in message_analyses:
            if msg.sentiment.label == "positive" and msg.sentiment.score > 0.7:
                indicators.append("high_satisfaction")
            elif "gratitude" in msg.emotions:
                indicators.append("appreciation")
            elif msg.sentiment.label == "negative" and msg.sentiment.score > 0.7:
                indicators.append("dissatisfaction")

        return indicators

    def _score_to_quality_level(self, score: float) -> str:
        """Convert score to quality level"""
        if score > 0.5:
            return "excellent"
        elif score > 0.1:
            return "good"
        elif score > -0.1:
            return "fair"
        else:
            return "poor"

    def _assess_professionalism(self, message_analyses: List[MessageAnalysis]) -> str:
        """Assess professionalism level"""
        # Simple heuristic based on language and tone
        professional_indicators = 0
        total_messages = len(message_analyses)

        for msg in message_analyses:
            text_lower = msg.text.lower()
            if any(word in text_lower for word in ['xin chào', 'hello', 'cảm ơn', 'thank you']):
                professional_indicators += 1
            if '!' in msg.text and msg.text.count('!') > 2:
                professional_indicators -= 0.5

        ratio = professional_indicators / total_messages if total_messages > 0 else 0

        if ratio > 0.5:
            return "high"
        elif ratio > 0.2:
            return "medium"
        else:
            return "low"

    def _calculate_empathy_score(self, message_analyses: List[MessageAnalysis]) -> float:
        """Calculate empathy score"""
        empathy_count = 0
        total_messages = len(message_analyses)

        for msg in message_analyses:
            if "empathy" in msg.emotions:
                empathy_count += 1
            if any(word in msg.text.lower() for word in ['hiểu', 'understand', 'cảm thông']):
                empathy_count += 0.5

        return min(empathy_count / total_messages if total_messages > 0 else 0, 1.0)

    def _assess_solution_effectiveness(self, message_analyses: List[MessageAnalysis]) -> float:
        """Assess the effectiveness of solutions provided"""
        solution_messages = [msg for msg in message_analyses if msg.intent == "solution"]

        if not solution_messages:
            return 0.0

        # Check if solutions led to positive sentiment
        effectiveness_score = 0
        for i, msg in enumerate(message_analyses):
            if msg.intent == "solution":
                # Check sentiment of following messages
                following_messages = message_analyses[i+1:i+3]
                if following_messages:
                    avg_following_sentiment = sum(
                        self._sentiment_to_score(m.sentiment) for m in following_messages
                    ) / len(following_messages)
                    if avg_following_sentiment > 0:
                        effectiveness_score += 1

        return min(effectiveness_score / len(solution_messages), 1.0)

    def _estimate_customer_satisfaction(
        self,
        participants: Dict[str, ParticipantSentiment],
        escalation_analysis: EscalationAnalysis
    ) -> float:
        """Estimate customer satisfaction score"""
        # Find customer participant
        customer_participants = []
        for speaker, sentiment in participants.items():
            if speaker in ['customer', 'user', 'client']:
                customer_participants.append(sentiment)

        if not customer_participants:
            # Use first participant as proxy
            customer_participants = list(participants.values())[:1]

        if not customer_participants:
            return 0.5

        customer = customer_participants[0]
        base_score = (customer.overall_score + 1) / 2  # Convert -1,1 to 0,1

        # Adjust based on escalation
        if escalation_analysis.escalation_level == "none":
            base_score += 0.1
        elif escalation_analysis.escalation_level in ["high", "critical"]:
            base_score -= 0.2

        return max(0, min(1, base_score))

    def _analyze_attitude(
        self,
        messages: List[MessageAnalysis],
        context: ConversationContext
    ) -> str:
        """Analyze participant attitude"""
        sentiment_scores = [self._sentiment_to_score(msg.sentiment) for msg in messages]
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)

        # Analyze cooperation indicators
        cooperation_indicators = 0
        for msg in messages:
            if msg.intent in ["acknowledgment", "solution"]:
                cooperation_indicators += 1
            if any(word in msg.text.lower() for word in ['cảm ơn', 'được', 'ok']):
                cooperation_indicators += 0.5

        cooperation_ratio = cooperation_indicators / len(messages)

        if avg_sentiment < -0.3 and cooperation_ratio < 0.2:
            return "hostile"
        elif avg_sentiment < -0.1:
            return "frustrated"
        elif cooperation_ratio > 0.5:
            return "cooperative"
        else:
            return "neutral"

    def _analyze_communication_style(self, messages: List[MessageAnalysis]) -> str:
        """Analyze communication style"""
        total_length = sum(len(msg.text) for msg in messages)
        avg_length = total_length / len(messages)

        # Count formal indicators
        formal_count = 0
        for msg in messages:
            if any(word in msg.text.lower() for word in ['xin chào', 'cảm ơn', 'trân trọng']):
                formal_count += 1

        formality_ratio = formal_count / len(messages)

        if formality_ratio > 0.3:
            return "formal"
        elif avg_length > 50:
            return "detailed"
        elif avg_length < 20:
            return "brief"
        else:
            return "casual"

    def _analyze_engagement_level(self, messages: List[MessageAnalysis]) -> str:
        """Analyze engagement level"""
        avg_length = sum(len(msg.text) for msg in messages) / len(messages)
        question_ratio = sum(1 for msg in messages if '?' in msg.text) / len(messages)

        if avg_length > 80 or question_ratio > 0.3:
            return "high"
        elif avg_length > 30:
            return "medium"
        else:
            return "low"

    def _calculate_detailed_scores(self, overall_score: float, confidence: float) -> Dict[str, float]:
        """Calculate detailed sentiment scores"""
        if overall_score > 0.15:
            pos_prob = confidence
            neg_prob = (1 - confidence) * 0.3
            neu_prob = 1 - pos_prob - neg_prob
        elif overall_score < -0.15:
            neg_prob = confidence
            pos_prob = (1 - confidence) * 0.3
            neu_prob = 1 - pos_prob - neg_prob
        else:
            neu_prob = 0.6
            pos_prob = neg_prob = 0.2

        # Normalize
        total = pos_prob + neg_prob + neu_prob
        return {
            "positive": round(pos_prob / total, 4),
            "negative": round(neg_prob / total, 4),
            "neutral": round(neu_prob / total, 4)
        }
