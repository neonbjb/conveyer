import math
import unittest
from time import sleep, time

import torch

from conveyer.core import Range, Conveyer, Partition, Collate, Map, Join
from multi import DistributedBatchedMap


def mapping_fn(x):
    return x * 2


def sleeping_mapping_fn(x):
    sleep(1)
    return x


class TestDistributedBatchedMap(unittest.TestCase):

    def test_basic_functionality(self):
        """Test if the mapping function is applied correctly."""
        dbm = Conveyer([Range(10, repeating=False), DistributedBatchedMap(mapping_fn, n_workers=2, buffer_size=5)])
        results = [dbm() for _ in range(10)]
        dbm.close()
        expected_results = [mapping_fn(i) for i in range(10)]
        self.assertEqual(set(results), set(expected_results))

    def test_multiple_elements(self):
        """Test if the class can process multiple elements."""
        dbm = Conveyer([Range(10, repeating=False), DistributedBatchedMap(mapping_fn, n_workers=2, buffer_size=5)])
        first_result = dbm()
        second_result = dbm()
        dbm.close()
        self.assertIsNotNone(first_result)
        self.assertIsNotNone(second_result)

    def test_constrained_functionality(self):
        """ A distributed batch map where the worker_fn has a sleep built in. """
        dbm = Conveyer([Range(100, repeating=False), DistributedBatchedMap(sleeping_mapping_fn, n_workers=50, timeout=10.0, buffer_size=50)])
        start = time()
        results = [dbm() for _ in range(100)]
        elapsed = time() - start
        dbm.close()
        for i in range(100):
            assert i in results
        assert elapsed < 20.0  # It takes a few seconds to boot up the threads, but the total time should be much less than the sequential wait of 100 seconds.

    def test_pause_resume_like_conveyer(self):
        """ Saving and resuming the state of a conveyer. """
        dbm = Conveyer([Range(10, repeating=False), DistributedBatchedMap(mapping_fn, n_workers=2, buffer_size=5)])
        results = [dbm() for _ in range(5)]
        dbm2 = Conveyer([Range(10, repeating=False), DistributedBatchedMap(mapping_fn, n_workers=2, buffer_size=5)])
        dbm2.set_state(dbm.state())
        results.extend([dbm2() for _ in range(5)])
        dbm.close()
        dbm2.close()
        expected_results = [mapping_fn(i) for i in range(10)]
        self.assertEqual(set(results), set(expected_results))

    def test_pause_resume_unlike_conveyer(self):
        """ Saving and resuming state of a conveyer when the new one has a different n_worker and buffer_size."""
        dbm = Conveyer([Range(10, repeating=False), DistributedBatchedMap(mapping_fn, n_workers=2, buffer_size=5)])
        results = [dbm() for _ in range(5)]
        dbm2 = Conveyer([Range(10, repeating=False), DistributedBatchedMap(mapping_fn, n_workers=4, buffer_size=9)])
        dbm2.set_state(dbm.state())
        results.extend([dbm2() for _ in range(5)])
        dbm.close()
        dbm2.close()
        expected_results = [mapping_fn(i) for i in range(10)]
        self.assertEqual(set(results), set(expected_results))

    def test_shard(self):
        """ Verifies sharded iteration works as expected with both a like-shard and a dis-like shard. """
        conveyer1 = Conveyer([
            Range(100, repeating=False),
            DistributedBatchedMap(mapping_fn, n_workers=2, buffer_size=7),
            Partition(rank=0, n_ranks=2)
        ])
        conveyer1_0 = Conveyer([
            Range(100, repeating=False),
            DistributedBatchedMap(mapping_fn, n_workers=2, buffer_size=7),
            Partition(rank=0, n_ranks=2)
        ])
        conveyer2 = Conveyer([
            Range(100, repeating=False),
            DistributedBatchedMap(mapping_fn, n_workers=2, buffer_size=7),
            Partition(rank=1, n_ranks=2)
        ])
        c1_elements, c1_t_elements, c2_elements, elements = [], [], [], []
        try:
            for _ in range(50):
                c1_1, c1_2, c2_1 = conveyer1(), conveyer1_0(), conveyer2()
                c1_elements.append(c1_1)
                c1_t_elements.append(c1_2)
                c2_elements.append(c2_1)
                elements.append(c1_1)
                elements.append(c2_1)
        except:
            conveyer1.close()
            conveyer1_0.close()
            conveyer2.close()
            raise
        conveyer1.close()
        conveyer1_0.close()
        conveyer2.close()
        self.assertEqual(set(elements), set([mapping_fn(i) for i in range(100)]))
        self.assertEqual(set(c1_elements), set(c1_t_elements))
        self.assertEqual(len(set(c1_elements).intersection(set(c2_elements))), 0)

    def test_shard_save_resume(self):
        """ Verifies that sharded iteration saves and resumes as expected. """
        conveyer1 = Conveyer([
            Range(10, repeating=False),
            DistributedBatchedMap(mapping_fn, n_workers=2, buffer_size=2),
            Partition(rank=0, n_ranks=2)
        ])
        conveyer2 = Conveyer([
            Range(10, repeating=False),
            DistributedBatchedMap(mapping_fn, n_workers=2, buffer_size=2),
            Partition(rank=1, n_ranks=2)
        ])
        elements = []
        for i in range(3):
            elements.append(conveyer1())
            elements.append(conveyer2())
        conveyer1.close()
        conveyer2.close()

        state = conveyer1.state()
        conveyer1 = Conveyer([
            Range(10, repeating=False),
            DistributedBatchedMap(mapping_fn, n_workers=2, buffer_size=2),
            Partition(rank=0, n_ranks=2)
        ])
        conveyer2 = Conveyer([
            Range(10, repeating=False),
            DistributedBatchedMap(mapping_fn, n_workers=2, buffer_size=2),
            Partition(rank=1, n_ranks=2)
        ])
        conveyer1.set_state(state)
        conveyer2.set_state(state)
        for i in range(2):
            elements.append(conveyer1())
            elements.append(conveyer2())

        conveyer1.close()
        conveyer2.close()
        self.assertEqual(set(elements), set([mapping_fn(i) for i in range(10)]))

    def test_batching(self):
        """ Tests that batch collation works in normal circumstances """
        dbm = Conveyer([Range(12, repeating=False),
                        Map(lambda x: {"val": x}),
                        Collate(batch_size=4)])
        results = [dbm() for _ in range(3)]
        dbm.close()
        results = torch.cat([r['val'] for r in results]).tolist()
        expected_results = [i for i in range(12)]
        self.assertEqual(set(results), set(expected_results))

    def test_dissimilar_batching(self):
        """ Tests that batch collation works when the corpus size is not a multiple of the batch size """
        dbm = Conveyer([Range(10, repeating=False),
                        Map(lambda x: {"val": x}),
                        Collate(batch_size=4)])
        results = [dbm() for _ in range(3)]
        dbm.close()
        results = torch.cat([r['val'] for r in results]).tolist()
        expected_results = [i for i in range(10)]
        self.assertEqual(set(results), set(expected_results))

    def test_batched_past_end(self):
        """ Tests that batch collation works and throws the correct exception when the corpus is exhausted """
        c = Conveyer([Range(15, repeating=False),
                      Map(lambda x: {"val": x}),
                      Collate(batch_size=4)])
        [c() for _ in range(4)]
        try:
            c()
        except StopIteration:
            pass

    def test_batched_multimap_saving(self):
        """ Tests collation+multimap+stage persistence """
        dbm = Conveyer([Range(100, repeating=False),
                        DistributedBatchedMap(mapping_fn, n_workers=4, buffer_size=3, timeout=10),
                        Map(lambda x: {"val": x}),
                        Collate(batch_size=8)])
        results1 = [dbm() for _ in range(40 // 8)]
        state = dbm.state()
        dbm.close()

        dbm = Conveyer([Range(100, repeating=False),
                        DistributedBatchedMap(mapping_fn, n_workers=4, buffer_size=3, timeout=10),
                        Map(lambda x: {"val": x}),
                        Collate(batch_size=8)])
        dbm.set_state(state)
        results2 = [dbm() for _ in range(math.ceil(60 / 8))]
        dbm.close()

        results = results1 + results2
        results = torch.cat([r['val'] for r in results]).tolist()
        expected_results = [mapping_fn(i) for i in range(100)]
        self.assertEqual(set(results), set(expected_results))

    def test_join(self):
        c1 = Conveyer([Range(10, repeating=True),
                       Map(lambda x: {"val": x}),
                       Collate(batch_size=4)])
        c2 = Conveyer([Range(10, repeating=True),
                       Map(lambda x: {"val": x + 10}),
                       Collate(batch_size=4)])
        joined = Join([(c1, 1.0), (c2, 1.0)])
        results = [joined() for _ in range(100)]
        results = torch.cat([r['val'] for r in results]).tolist()
        expected_results = [i for i in range(10)] + [(i+10) for i in range(10)]
        self.assertEqual(set(results), set(expected_results))


if __name__ == '__main__':
    unittest.main()
