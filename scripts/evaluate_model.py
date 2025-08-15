"""
Enhanced sentiment analysis testing and evaluation script
Tests the improved rule-based model against challenging cases
"""

import asyncio
import time
import logging
from typing import List, Dict, Any
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models.enhanced_rule_based import EnhancedRuleBasedModel
from data.samples.test_data import TEST_CASES, EDGE_CASES, BENCHMARK_TEXTS

logger = logging.getLogger(__name__)

class SentimentEvaluator:
    """Evaluates sentiment analysis model performance"""

    def __init__(self):
        self.model = None
        self.results = {
            'total_tests': 0,
            'correct_predictions': 0,
            'accuracy': 0.0,
            'by_difficulty': {},
            'failed_cases': [],
            'performance_metrics': {}
        }

    async def initialize(self):
        """Initialize the enhanced model"""
        print("🚀 Initializing Enhanced Rule-Based Model...")
        self.model = EnhancedRuleBasedModel()
        await self.model.load()
        print("✅ Model loaded successfully!")

    async def run_comprehensive_test(self):
        """Run comprehensive accuracy and performance tests"""
        print("\n" + "="*60)
        print("🧪 COMPREHENSIVE SENTIMENT ANALYSIS EVALUATION")
        print("="*60)

        # Test accuracy
        await self._test_accuracy()

        # Test edge cases
        await self._test_edge_cases()

        # Performance benchmark
        await self._test_performance()

        # Generate report
        self._generate_report()

    async def _test_accuracy(self):
        """Test accuracy on challenging cases"""
        print("\n📊 ACCURACY TESTING")
        print("-" * 40)

        correct = 0
        total = len(TEST_CASES)
        difficulty_stats = {}

        for i, case in enumerate(TEST_CASES):
            text = case['text']
            expected = case['expected']
            difficulty = case['difficulty']

            # Get prediction
            prediction = await self.model.predict(text)
            predicted_label = prediction.label
            is_correct = predicted_label == expected

            if is_correct:
                correct += 1
                status = "✅"
            else:
                status = "❌"
                self.results['failed_cases'].append({
                    'text': text,
                    'expected': expected,
                    'predicted': predicted_label,
                    'confidence': prediction.score,
                    'difficulty': difficulty
                })

            # Track by difficulty
            if difficulty not in difficulty_stats:
                difficulty_stats[difficulty] = {'correct': 0, 'total': 0}
            difficulty_stats[difficulty]['total'] += 1
            if is_correct:
                difficulty_stats[difficulty]['correct'] += 1

            print(f"{i+1:2d}. {status} [{difficulty:8s}] {text[:50]:<50} | Expected: {expected:8s} | Got: {predicted_label:8s} | Conf: {prediction.score:.3f}")

        accuracy = correct / total
        self.results.update({
            'total_tests': total,
            'correct_predictions': correct,
            'accuracy': accuracy,
            'by_difficulty': {
                diff: stats['correct'] / stats['total']
                for diff, stats in difficulty_stats.items()
            }
        })

        print(f"\n📈 ACCURACY SUMMARY:")
        print(f"   Overall: {correct}/{total} = {accuracy:.2%}")
        for difficulty, acc in self.results['by_difficulty'].items():
            print(f"   {difficulty:12s}: {acc:.2%}")

    async def _test_edge_cases(self):
        """Test edge cases and robustness"""
        print("\n🔧 EDGE CASE TESTING")
        print("-" * 40)

        edge_correct = 0
        edge_total = len(EDGE_CASES)

        for i, case in enumerate(EDGE_CASES):
            text = case['text']
            expected = case['expected']
            description = case['description']

            try:
                prediction = await self.model.predict(text)
                predicted_label = prediction.label
                is_correct = predicted_label == expected

                if is_correct:
                    edge_correct += 1
                    status = "✅"
                else:
                    status = "❌"

                print(f"{i+1}. {status} [{description:20s}] '{text[:30]}' | Expected: {expected} | Got: {predicted_label}")

            except Exception as e:
                print(f"{i+1}. 💥 [{description:20s}] ERROR: {str(e)}")

        edge_accuracy = edge_correct / edge_total if edge_total > 0 else 0
        print(f"\n🛡️  ROBUSTNESS: {edge_correct}/{edge_total} = {edge_accuracy:.2%}")

    async def _test_performance(self):
        """Test performance and speed"""
        print("\n⚡ PERFORMANCE TESTING")
        print("-" * 40)

        # Single prediction performance
        text = "Tôi rất hài lòng với sản phẩm này"

        start_time = time.time()
        for _ in range(100):
            await self.model.predict(text)
        single_avg_time = (time.time() - start_time) / 100

        # Batch prediction performance
        start_time = time.time()
        await self.model.predict_batch(BENCHMARK_TEXTS)
        batch_time = time.time() - start_time
        batch_avg_time = batch_time / len(BENCHMARK_TEXTS)

        # Memory usage test (simplified)
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024

        self.results['performance_metrics'] = {
            'single_prediction_avg_ms': single_avg_time * 1000,
            'batch_prediction_avg_ms': batch_avg_time * 1000,
            'memory_usage_mb': memory_mb
        }

        print(f"   Single prediction: {single_avg_time*1000:.2f} ms")
        print(f"   Batch avg per text: {batch_avg_time*1000:.2f} ms")
        print(f"   Memory usage: {memory_mb:.1f} MB")

    def _generate_report(self):
        """Generate comprehensive evaluation report"""
        print("\n" + "="*60)
        print("📋 FINAL EVALUATION REPORT")
        print("="*60)

        accuracy = self.results['accuracy']

        # Overall grade
        if accuracy >= 0.95:
            grade = "🏆 EXCELLENT"
            grade_desc = "Near-perfect accuracy achieved!"
        elif accuracy >= 0.90:
            grade = "🥇 OUTSTANDING"
            grade_desc = "Exceptional performance!"
        elif accuracy >= 0.85:
            grade = "🥈 VERY GOOD"
            grade_desc = "Strong performance with room for improvement"
        elif accuracy >= 0.80:
            grade = "🥉 GOOD"
            grade_desc = "Good baseline performance"
        else:
            grade = "⚠️  NEEDS IMPROVEMENT"
            grade_desc = "Requires significant optimization"

        print(f"\n🎯 OVERALL PERFORMANCE: {grade}")
        print(f"   {grade_desc}")
        print(f"\n📊 DETAILED METRICS:")
        print(f"   • Total Accuracy: {accuracy:.2%}")
        print(f"   • Correct Predictions: {self.results['correct_predictions']}/{self.results['total_tests']}")

        print(f"\n📈 ACCURACY BY DIFFICULTY:")
        for difficulty, acc in sorted(self.results['by_difficulty'].items()):
            if acc >= 0.9:
                status = "🟢"
            elif acc >= 0.7:
                status = "🟡"
            else:
                status = "🔴"
            print(f"   {status} {difficulty:12s}: {acc:.2%}")

        print(f"\n⚡ PERFORMANCE METRICS:")
        perf = self.results['performance_metrics']
        print(f"   • Single prediction: {perf['single_prediction_avg_ms']:.2f} ms")
        print(f"   • Batch processing: {perf['batch_prediction_avg_ms']:.2f} ms/text")
        print(f"   • Memory usage: {perf['memory_usage_mb']:.1f} MB")

        if self.results['failed_cases']:
            print(f"\n❌ FAILED CASES ({len(self.results['failed_cases'])}):")
            for i, case in enumerate(self.results['failed_cases'][:5]):  # Show first 5
                print(f"   {i+1}. [{case['difficulty']}] {case['text'][:40]}...")
                print(f"      Expected: {case['expected']} | Got: {case['predicted']} | Conf: {case['confidence']:.3f}")
            if len(self.results['failed_cases']) > 5:
                print(f"   ... and {len(self.results['failed_cases']) - 5} more")

        print(f"\n💡 RECOMMENDATIONS:")
        if accuracy < 0.85:
            print("   • Expand sentiment lexicon with more domain-specific terms")
            print("   • Improve sarcasm and irony detection patterns")
            print("   • Add more contextual rules for complex sentences")
        elif accuracy < 0.95:
            print("   • Fine-tune pattern weights and thresholds")
            print("   • Add more temporal and conditional patterns")
            print("   • Enhance emoji and slang coverage")
        else:
            print("   • Model performing excellently!")
            print("   • Consider adding domain-specific adaptations")
            print("   • Monitor performance on real-world data")

async def main():
    """Main evaluation function"""
    evaluator = SentimentEvaluator()

    try:
        await evaluator.initialize()
        await evaluator.run_comprehensive_test()

    except Exception as e:
        print(f"💥 Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.WARNING)

    # Run evaluation
    asyncio.run(main())
