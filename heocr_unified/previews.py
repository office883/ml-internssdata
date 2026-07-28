from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageOps

from .metadata import write_json_atomic
from .unicode_utils import normalize_label_strict


@dataclass(frozen=True)
class PreviewCandidate:
    category: str
    sample_id: str
    text: str
    image_bytes: bytes
    metadata: dict[str, Any]


def _sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def write_contact_sheet(
    stem: str | Path,
    *,
    title: str,
    candidates: Iterable[PreviewCandidate],
    columns: int = 4,
) -> dict[str, Any]:
    stem=Path(stem)
    stem.parent.mkdir(parents=True,exist_ok=True)
    rows=sorted(candidates,key=lambda item:(item.category,item.sample_id))
    if not rows:
        raise ValueError("contact sheet has no candidates")
    tile_w,tile_h=340,180
    header=36
    count_rows=(len(rows)+columns-1)//columns
    canvas=Image.new("RGB",(columns*tile_w,header+count_rows*tile_h),(245,245,245))
    draw=ImageDraw.Draw(canvas)
    draw.text((10,10),title,fill=(0,0,0))
    sidecar=[]
    for index,row in enumerate(rows):
        col=index%columns; r=index//columns
        x=col*tile_w; y=header+r*tile_h
        with Image.open(io.BytesIO(row.image_bytes)) as source:
            source.load()
            thumb=ImageOps.contain(source.convert("RGB"),(tile_w-20,120),Image.Resampling.LANCZOS)
        canvas.paste(thumb,(x+10,y+8))
        caption=f"{row.category[:34]} | {row.sample_id[:20]}"
        draw.text((x+10,y+134),caption,fill=(10,10,10))
        sidecar.append({
            "category":row.category,"sample_id":row.sample_id,"text":row.text,
            "metadata":row.metadata,
        })
    image_path=stem.with_suffix(".png")
    sidecar_path=stem.with_suffix(".json")
    canvas.save(image_path,"PNG",optimize=False,compress_level=9)
    sidecar_path.write_text(json.dumps({"title":title,"samples":sidecar},ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return {
        "title":title,"samples":len(rows),"image":image_path.name,"sidecar":sidecar_path.name,
        "image_sha256":_sha256(image_path),"sidecar_sha256":_sha256(sidecar_path),
    }


def _candidate_rank(dimension: str, category: str, sample_id: str) -> str:
    return hashlib.sha256(f"{dimension}\x1f{category}\x1f{sample_id}".encode()).hexdigest()


def generate_previews(output_root: str | Path, *, max_per_sheet: int = 64) -> dict[str, Any]:
    import pyarrow.parquet as pq

    root=Path(output_root)
    preview_dir=root/"previews"
    if preview_dir.exists():
        for path in preview_dir.glob("*"):
            if path.is_file(): path.unlink()
    preview_dir.mkdir(parents=True,exist_ok=True)
    winners: dict[tuple[str,str],tuple[str,PreviewCandidate]]={}
    config_splits:set[str]=set()
    for path in sorted((root/"data").rglob("*.parquet")):
        rel=path.relative_to(root)
        config_name=rel.parts[1]; path_split=rel.parts[2]
        parquet=pq.ParquetFile(path)
        columns=[name for name in [
            "sample_id","image","text","split","modality","data_tier","source_repo",
            "font_family","augmentation_json","task",
        ] if name in parquet.schema_arrow.names]
        for batch in parquet.iter_batches(batch_size=256,columns=columns):
            for row in batch.to_pylist():
                image=row["image"]["bytes"]
                label=normalize_label_strict(row["text"])
                try: aug=json.loads(row.get("augmentation_json") or "{}")
                except json.JSONDecodeError: aug={}
                profile=str(aug.get("profile") or (aug.get("augmentation") or {}).get("profile") or "none")
                layout=str(aug.get("layout") or "none")
                categories={
                    "config_split":f"{config_name}|{path_split}",
                    "source_repo":str(row.get("source_repo") or "unknown"),
                    "modality":str(row.get("modality") or "unknown"),
                    "data_tier":str(row.get("data_tier") or "unknown"),
                    "augmentation_profile":profile,
                    "font_family":str(row.get("font_family") or "unspecified"),
                    "task":str(row.get("task") or "unknown"),
                }
                if layout!="none": categories["page_layout"]=layout
                if label.mixed_bidi: categories["feature"]="mixed_bidi"
                elif label.combining_marks: categories["feature"]="combining_marks"
                elif label.digits: categories["feature"]="digits"
                config_splits.add(categories["config_split"])
                for dimension,category in categories.items():
                    candidate=PreviewCandidate(
                        category=category,sample_id=row["sample_id"],text=row["text"],image_bytes=image,
                        metadata={"config":config_name,"split":path_split,"profile":profile,"layout":layout},
                    )
                    rank=_candidate_rank(dimension,category,row["sample_id"])
                    key=(dimension,category)
                    if key not in winners or rank<winners[key][0]: winners[key]=(rank,candidate)
    sheets=[]
    dimensions=sorted({dimension for dimension,_ in winners})
    for dimension in dimensions:
        rows=[value[1] for key,value in winners.items() if key[0]==dimension]
        rows.sort(key=lambda item:item.category)
        for chunk_index in range(0,len(rows),max_per_sheet):
            chunk=rows[chunk_index:chunk_index+max_per_sheet]
            suffix=chunk_index//max_per_sheet
            sheets.append(write_contact_sheet(
                preview_dir/f"{dimension}-{suffix:02d}",title=f"{dimension} [{suffix}]",candidates=chunk,
            ))
    inventory={
        "category_count":len(winners),"dimensions":dimensions,"sheets":sheets,
        "config_splits":sorted(config_splits),
        "categories":{
            dimension:sorted(category for dim,category in winners if dim==dimension)
            for dimension in dimensions
        },
    }
    write_json_atomic(preview_dir/"PREVIEW_INVENTORY.json",inventory)
    return inventory
