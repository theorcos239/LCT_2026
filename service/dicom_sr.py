# -*- coding: utf-8 -*-
"""DICOM SR с текстовым описанием нарушений качества (ТЗ 2.6, доп. функционал).

Формат — **Basic Text SR** (SOP Class `1.2.840.10008.5.1.4.1.1.88.11`),
самый простой вариант Structured Report: корневой контейнер с одним текстовым
узлом (вывод на русском, тот же текст, что в `violation_description`) и
ссылкой на исходное изображение. Полный TID 1500 («Measurement Report» с
кодами по каждому измерению) для этой задачи избыточен: он нужен, когда
числа должны быть машиночитаемы бланком стороннего PACS, а числа уже есть в
JSON (`--details`) — здесь нужен человекочитаемый вывод рядом со снимком.

    from service.dicom_sr import build_sr
    sr = build_sr(source_ds, result, VIOLATIONS, REGION_RU)
    sr.save_as('report.sr.dcm', enforce_file_format=True)
"""
from __future__ import annotations

import datetime as dt

import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

SR_SOP_CLASS = '1.2.840.10008.5.1.4.1.1.88.11'   # Basic Text SR Storage

# Код DCM «Findings» — тот же, что использует корневой контейнер TID 1500.
_FINDINGS_CODE = ('121070', 'DCM', 'Findings')
_REFERENCED_IMAGE_CODE = ('121191', 'DCM', 'Referenced Image')

# Теги пациента/исследования копируются как есть: SR описывает тот же
# эпизод, что и исходный снимок, и должен разделять его идентификаторы,
# иначе PACS не свяжет отчёт со снимком.
_PATIENT_STUDY_TAGS = (
    'PatientName', 'PatientID', 'PatientBirthDate', 'PatientSex',
    'StudyInstanceUID', 'StudyDate', 'StudyTime', 'StudyID',
    'AccessionNumber', 'ReferringPhysicianName',
)


def _code_item(value: str, scheme: str, meaning: str) -> Dataset:
    d = Dataset()
    d.CodeValue = value
    d.CodingSchemeDesignator = scheme
    d.CodeMeaning = meaning
    return d


def _text_item(code: tuple[str, str, str], text: str) -> Dataset:
    d = Dataset()
    d.RelationshipType = 'CONTAINS'
    d.ValueType = 'TEXT'
    d.ConceptNameCodeSequence = Sequence([_code_item(*code)])
    d.TextValue = text
    return d


def _image_item(sop_class_uid: str, sop_instance_uid: str) -> Dataset:
    ref = Dataset()
    ref.ReferencedSOPClassUID = sop_class_uid
    ref.ReferencedSOPInstanceUID = sop_instance_uid
    d = Dataset()
    d.RelationshipType = 'CONTAINS'
    d.ValueType = 'IMAGE'
    d.ConceptNameCodeSequence = Sequence([_code_item(*_REFERENCED_IMAGE_CODE)])
    d.ReferencedSOPSequence = Sequence([ref])
    return d


def build_report_text(result: dict, violation_texts: dict[str, str],
                      region_texts: dict[str, str] | None = None) -> str:
    """Тот же человекочитаемый вывод, что уходит в `violation_description`."""
    code = result.get('anatomical_region', 'unknown')
    region = (region_texts or {}).get(code) or code
    violations = result.get('violations') or result.get('violation_type', '')
    if isinstance(violations, str):
        violations = [v for v in violations.split(';') if v]
    lines = [f'Область: {region}']
    if not violations:
        lines.append('Нарушений качества не обнаружено.')
    else:
        lines.append('Обнаруженные нарушения:')
        lines += [f'  - {violation_texts.get(v, v)}' for v in violations]
    flags = result.get('flags') or []
    if isinstance(flags, str):
        flags = [f for f in flags.split(';') if f]
    if flags:
        lines.append('Флаги: ' + '; '.join(flags))
    return '\n'.join(lines)


def build_sr(source: Dataset, result: dict, violation_texts: dict[str, str],
             region_texts: dict[str, str] | None = None,
             *, series_number: int = 900, instance_number: int = 1) -> FileDataset:
    """DICOM SR по результату анализа одного кадра.

    `source` — датасет исходного DICOM (нужны UID и теги пациента/
    исследования для ссылки), `result` — вывод `Analyzer.analyze_pixels`
    или строка отчёта, `violation_texts` — `pipeline.VIOLATIONS`,
    `region_texts` — `pipeline.REGION_RU`.
    """
    now = dt.datetime.now()
    date_str, time_str = now.strftime('%Y%m%d'), now.strftime('%H%M%S.%f')

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SR_SOP_CLASS
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b'\x00' * 128)
    # UTF-8: находки и ФИО пациента — кириллица, ISO_IR 100 их не кодирует.
    ds.SpecificCharacterSet = 'ISO_IR 192'

    for tag in _PATIENT_STUDY_TAGS:
        setattr(ds, tag, source.get(tag, ''))

    ds.SeriesInstanceUID = generate_uid()
    ds.SeriesNumber = series_number
    ds.InstanceNumber = instance_number
    ds.Modality = 'SR'
    ds.SeriesDescription = 'DXA Quality Control Report'
    ds.Manufacturer = 'dxa-qc'

    ds.SOPClassUID = SR_SOP_CLASS
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.ContentDate = ds.InstanceCreationDate = date_str
    ds.ContentTime = ds.InstanceCreationTime = time_str

    # Type 2: MVP отчёта не проходит верификацию врачом.
    ds.CompletionFlag = 'COMPLETE'
    ds.VerificationFlag = 'UNVERIFIED'

    content = [_text_item(_FINDINGS_CODE, build_report_text(result, violation_texts, region_texts))]
    sop_class, sop_inst = str(source.get('SOPClassUID', '') or ''), \
        str(source.get('SOPInstanceUID', '') or '')
    if sop_class and sop_inst:
        content.append(_image_item(sop_class, sop_inst))

    # Корневой узел дерева контента — это сам документ, отдельного объекта не
    # заводится; у него, в отличие от детей, нет RelationshipType (Type 1C).
    ds.ValueType = 'CONTAINER'
    ds.ConceptNameCodeSequence = Sequence([_code_item(*_FINDINGS_CODE)])
    ds.ContinuityOfContent = 'SEPARATE'
    ds.ContentSequence = Sequence(content)

    return ds
