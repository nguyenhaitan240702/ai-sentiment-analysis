"""
Test script for conversation sentiment analysis
Tests the new conversation endpoint with various scenarios
"""

import asyncio
import json
import sys
import os
from datetime import datetime
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apps.api.schemas.request import (
    ConversationSentimentRequest,
    ConversationMessage,
    ConversationContext,
    ConversationAnalysisOptions,
    ConversationParticipant
)
from apps.api.services.conversation_service import ConversationSentimentService
from apps.api.services.sentiment_service import SentimentService
from core.models.enhanced_rule_based import EnhancedRuleBasedModel

class ConversationTestRunner:
    """Test runner for conversation sentiment analysis"""

    def __init__(self):
        self.sentiment_service = None
        self.conversation_service = None

        # Test conversations
        self.test_conversations = [
            {
                "name": "Customer Service Resolution",
                "conversation": [
                    {"speaker": "customer", "text": "Sản phẩm này bị lỗi hoài, tôi rất bực mình!"},
                    {"speaker": "agent", "text": "Tôi rất hiểu cảm giác của anh. Để tôi hỗ trợ anh giải quyết vấn đề này nhé."},
                    {"speaker": "customer", "text": "Được, nhưng tôi đã gọi nhiều lần rồi mà chưa ai giải quyết được."},
                    {"speaker": "agent", "text": "Tôi xin lỗi vì sự bất tiện này. Tôi sẽ xử lý ngay cho anh. Anh có thể cho tôi mã sản phẩm được không?"},
                    {"speaker": "customer", "text": "Mã sản phẩm là ABC123. Tôi mong là lần này sẽ được giải quyết."},
                    {"speaker": "agent", "text": "Tôi đã kiểm tra và sẽ gửi sản phẩm thay thế cho anh ngay hôm nay. Anh sẽ nhận được trong 24h."},
                    {"speaker": "customer", "text": "Cảm ơn, như vậy là tốt rồi. Tôi hài lòng với cách giải quyết này."}
                ],
                "context": {
                    "domain": "customer_service",
                    "conversation_type": "support_ticket",
                    "participants": {
                        "customer": {"role": "customer", "segment": "regular"},
                        "agent": {"role": "support_agent", "experience_level": "senior"}
                    }
                }
            },
            {
                "name": "Sales Conversation",
                "conversation": [
                    {"speaker": "prospect", "text": "Tôi đang quan tâm đến sản phẩm của các bạn."},
                    {"speaker": "sales", "text": "Cảm ơn anh đã quan tâm! Tôi có thể giới thiệu chi tiết về sản phẩm cho anh."},
                    {"speaker": "prospect", "text": "Giá cả thế nào? Có đắt không?"},
                    {"speaker": "sales", "text": "Với chất lượng và tính năng của sản phẩm, giá rất hợp lý. Chúng tôi có nhiều gói để anh lựa chọn."},
                    {"speaker": "prospect", "text": "Nghe có vẻ hay đấy. Tôi cần suy nghĩ thêm."},
                    {"speaker": "sales", "text": "Không vấn đề gì! Tôi sẽ gửi tài liệu chi tiết cho anh. Anh có thể liên hệ khi nào cần hỗ trợ."}
                ],
                "context": {
                    "domain": "sales",
                    "conversation_type": "sales_inquiry"
                }
            },
            {
                "name": "Escalated Complaint",
                "conversation": [
                    {"speaker": "customer", "text": "Dịch vụ của các bạn quá tệ! Tôi đã chờ 2 tiếng mà không được giải quyết!"},
                    {"speaker": "agent", "text": "Tôi hiểu anh rất bực mình. Để tôi xem thông tin..."},
                    {"speaker": "customer", "text": "Tôi không muốn chờ nữa! Đây là lần thứ 3 tôi gọi vào rồi!"},
                    {"speaker": "agent", "text": "Tôi thật sự xin lỗi về sự bất tiện này. Tôi sẽ chuyển cho supervisor để giải quyết ngay."},
                    {"speaker": "supervisor", "text": "Chào anh, tôi là supervisor. Tôi đã xem qua vấn đề và sẽ giải quyết ngay lập tức."},
                    {"speaker": "customer", "text": "Cuối cùng cũng có người có thẩm quyền! Tôi muốn được bồi thường."},
                    {"speaker": "supervisor", "text": "Hoàn toàn hợp lý. Chúng tôi sẽ hoàn tiền và tặng thêm voucher 500k để bù đắp."},
                    {"speaker": "customer", "text": "Được rồi, như vậy là ổn. Cảm ơn đã giải quyết nhanh chóng."}
                ],
                "context": {
                    "domain": "customer_service",
                    "conversation_type": "escalated_complaint"
                }
            }
        ]

    async def initialize(self):
        """Initialize services"""
        print("🚀 Initializing Conversation Sentiment Analysis Test...")

        # Initialize enhanced model
        model = EnhancedRuleBasedModel()
        await model.load()

        # Create mock model manager
        class MockModelManager:
            def __init__(self):
                self.model = model
                self.default_model = "enhanced_rule_based_vi"

            async def predict(self, text, language="vi"):
                return await self.model.predict(text, language)

        model_manager = MockModelManager()

        # Initialize sentiment service
        self.sentiment_service = SentimentService()
        await self.sentiment_service.initialize(model_manager)

        print("✅ Services initialized successfully!")

    async def test_conversation(self, test_case: Dict[str, Any]):
        """Test a single conversation"""
        print(f"\n📊 Testing: {test_case['name']}")
        print("-" * 50)

        # Create conversation messages
        messages = []
        for i, msg in enumerate(test_case['conversation']):
            message = ConversationMessage(
                message_id=f"msg_{i}",
                speaker=msg['speaker'],
                text=msg['text'],
                timestamp=datetime.now()
            )
            messages.append(message)

        # Create context
        context = None
        if 'context' in test_case:
            participants = {}
            for speaker, participant_data in test_case['context'].get('participants', {}).items():
                participants[speaker] = ConversationParticipant(**participant_data)

            context = ConversationContext(
                domain=test_case['context'].get('domain'),
                conversation_type=test_case['context'].get('conversation_type'),
                participants=participants
            )

        # Create request
        request = ConversationSentimentRequest(
            conversation=messages,
            context=context,
            analysis_options=ConversationAnalysisOptions(
                include_emotional_flow=True,
                include_participant_analysis=True,
                include_escalation_tracking=True,
                include_summary=True,
                granularity="detailed"
            )
        )

        # Analyze conversation
        try:
            result = await self.sentiment_service.analyze_conversation(request)

            # Print results
            self._print_results(result, test_case['name'])

            return True

        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _print_results(self, result, test_name):
        """Print formatted results"""
        print(f"🎯 Overall Sentiment: {result.overall_sentiment.label.upper()} (confidence: {result.overall_sentiment.score:.3f})")
        print(f"🗣️  Conversation Type: {result.conversation_type}")

        print(f"\n📈 Emotional Flow:")
        print(f"   Trajectory: {result.emotional_flow.trajectory}")
        print(f"   Stability: {result.emotional_flow.stability}")
        print(f"   Dominant Emotion: {result.emotional_flow.dominant_emotion}")
        if result.emotional_flow.turning_points:
            print(f"   Turning Points: {len(result.emotional_flow.turning_points)}")
            for tp in result.emotional_flow.turning_points:
                print(f"      • Message {tp.message_index}: {tp.from_sentiment} → {tp.to_sentiment} ({tp.trigger})")

        print(f"\n👥 Participants:")
        for speaker, analysis in result.participants.items():
            print(f"   {speaker.upper()}:")
            print(f"      Overall: {analysis.overall_sentiment} (score: {analysis.overall_score:.3f})")
            print(f"      Attitude: {analysis.attitude}")
            print(f"      Style: {analysis.communication_style}")
            print(f"      Engagement: {analysis.engagement_level}")
            if analysis.dominant_emotions:
                print(f"      Emotions: {', '.join(analysis.dominant_emotions)}")

        print(f"\n🔥 Escalation Analysis:")
        print(f"   Level: {result.escalation_analysis.escalation_level}")
        print(f"   Trend: {result.escalation_analysis.escalation_trend}")
        print(f"   Conflict Intensity: {result.escalation_analysis.conflict_intensity:.3f}")
        if result.escalation_analysis.resolution_indicators:
            print(f"   Resolution Signs: {', '.join(result.escalation_analysis.resolution_indicators)}")

        print(f"\n📋 Summary:")
        print(f"   Topics: {', '.join(result.conversation_summary.main_topics) if result.conversation_summary.main_topics else 'None detected'}")
        print(f"   Issues: {', '.join(result.conversation_summary.key_issues) if result.conversation_summary.key_issues else 'None detected'}")
        print(f"   Resolution: {result.conversation_summary.resolution_status}")
        print(f"   Outcome: {result.conversation_summary.conversation_outcome}")

        print(f"\n⭐ Quality Metrics:")
        print(f"   Communication: {result.conversation_quality.communication_quality}")
        print(f"   Professionalism: {result.conversation_quality.professionalism_level}")
        print(f"   Empathy: {result.conversation_quality.empathy_score:.3f}")
        print(f"   Solution Effectiveness: {result.conversation_quality.solution_effectiveness:.3f}")
        print(f"   Customer Satisfaction: {result.conversation_quality.customer_satisfaction_score:.3f}")

        print(f"\n⚡ Performance:")
        metadata = result.analysis_metadata
        print(f"   Processing Time: {metadata.get('processing_time_ms', 0)}ms")
        print(f"   Messages: {metadata.get('message_count', 0)}")
        print(f"   Participants: {metadata.get('participant_count', 0)}")

    async def run_all_tests(self):
        """Run all conversation tests"""
        await self.initialize()

        print("\n" + "="*60)
        print("🧪 CONVERSATION SENTIMENT ANALYSIS TESTS")
        print("="*60)

        passed = 0
        total = len(self.test_conversations)

        for test_case in self.test_conversations:
            success = await self.test_conversation(test_case)
            if success:
                passed += 1

        print("\n" + "="*60)
        print(f"📊 TEST SUMMARY: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        print("="*60)

        if passed == total:
            print("🎉 All tests passed! Conversation sentiment analysis is working correctly.")
        else:
            print(f"⚠️  {total - passed} test(s) failed. Please check the errors above.")

async def main():
    """Main test function"""
    runner = ConversationTestRunner()
    await runner.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())
