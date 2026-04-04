"""Concrete pipeline strategy that groups failures by signature before ranking."""

from __future__ import annotations

from collections import defaultdict

from ca.core.check_triage import (
    CheckTriageItem,
    CheckTriageRequest,
    CheckTriageResult,
    RankedCheckTriageItem,
    normalized_dimension_gaps,
)


class SignatureClusterCheckTriageStrategy:
    """Surface one representative per failure signature before duplicates."""

    name = "signature_cluster"
    design = "pipeline"

    def rank(self, request: CheckTriageRequest) -> CheckTriageResult:
        clusters: dict[str, list[tuple[dict[str, float], CheckTriageItem]]] = defaultdict(list)
        for item in request.failed_items:
            gaps = normalized_dimension_gaps(item)
            signature = str(item.metadata.get("signature") or (item.kind, tuple(gaps.keys())))
            clusters[signature].append((gaps, item))

        ordered_clusters = sorted(
            clusters.items(),
            key=lambda entry: (
                -max(sum(gaps.values()) for gaps, _ in entry[1]),
                -len(entry[1]),
                entry[0],
            ),
        )

        round_robin: list[tuple[dict[str, float], CheckTriageItem, str]] = []
        max_cluster_size = max((len(items) for _, items in ordered_clusters), default=0)
        for offset in range(max_cluster_size):
            for signature, items in ordered_clusters:
                if offset >= len(items):
                    continue
                sorted_items = sorted(
                    items,
                    key=lambda pair: (-sum(pair[0].values()), pair[1].check_id),
                )
                gaps, item = sorted_items[offset]
                round_robin.append((gaps, item, signature))

        ranked_items = []
        for index, (gaps, item, signature) in enumerate(round_robin, start=1):
            ranked_items.append(
                RankedCheckTriageItem(
                    check_id=item.check_id,
                    kind=item.kind,
                    rank=index,
                    severity_score=round(float(sum(gaps.values())), 6),
                    failed_dimensions=tuple(gaps.keys()),
                    reasons=item.reasons,
                    metrics=dict(item.metrics),
                    gate=dict(item.gate),
                    report_path=item.report_path,
                    json_path=item.json_path,
                    metadata={
                        **dict(item.metadata),
                        "cluster_signature": signature,
                    },
                )
            )
        return CheckTriageResult(
            ranked_items=tuple(ranked_items),
            strategy=self.name,
            design=self.design,
            metadata={"cluster_count": len(clusters)},
        )
