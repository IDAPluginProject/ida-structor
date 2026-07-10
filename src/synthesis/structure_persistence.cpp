/// @file structure_persistence.cpp
/// @brief Structure persistence implementation

#include <structor/structure_persistence.hpp>
#include <structor/naming.hpp>
#include <structor/optimized_algorithms.hpp>
#include <structor/persistence_invariants.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace structor {

namespace {

struct ExpectedUdtMember {
    qstring name;
    std::uint64_t bit_offset = 0;
    std::uint64_t bit_size = 0;
    bool is_bitfield = false;
    bool bitfield_encoding_valid = true;
    std::uint8_t bitfield_storage_bytes = 0;
    std::uint8_t bitfield_width = 0;
    bool bitfield_is_unsigned = false;
};

struct UdtLayoutExpectation {
    std::uint8_t pack_code = 0;
    std::uint32_t effective_alignment = 0;
    size_t size = 0;
    bool is_union = false;
    std::vector<ExpectedUdtMember> members;
};

struct BitCoverageRange {
    std::uint64_t begin = 0;
    std::uint64_t end = 0;
};

[[nodiscard]] bool validate_public_struct_shape(
    const SynthStruct& synth_struct) noexcept {
    if (synth_struct.name.empty() || synth_struct.size == 0 ||
        synth_struct.size > MAX_STRUCT_SIZE || synth_struct.fields.empty() ||
        synth_struct.fields.size() > MAX_FIELDS) {
        return false;
    }

    for (const auto& field : synth_struct.fields) {
        if (field.offset < 0 || field.size == 0) {
            return false;
        }
        const auto end = checked_interval_end(field.offset, field.size);
        if (!end.has_value() || *end < 0 ||
            static_cast<std::uint64_t>(*end) > synth_struct.size ||
            static_cast<std::uint64_t>(*end) > MAX_STRUCT_SIZE ||
            field.union_members.size() > MAX_FIELDS) {
            return false;
        }
        if (field.is_bitfield) {
            const std::uint64_t storage_bits =
                static_cast<std::uint64_t>(field.size) * 8;
            const std::uint64_t width = field.bit_size == 0
                ? storage_bits
                : field.bit_size;
            if (width == 0 || field.bit_offset >= storage_bits ||
                width > storage_bits - field.bit_offset) {
                return false;
            }
        }
    }
    return true;
}

[[nodiscard]] bool is_persistence_padding_member(const udm_t& member) noexcept {
    return member.is_gap() || member.name.find("__pad_") == 0;
}

[[nodiscard]] bool append_synth_field_coverage(
    std::vector<BitCoverageRange>& coverage, const SynthField& field) noexcept {
    if (field.is_padding) {
        return true;
    }
    if (field.offset < 0 || field.size == 0) {
        return false;
    }

    constexpr std::uint64_t kBitsPerByte = 8;
    const std::uint64_t offset = static_cast<std::uint64_t>(field.offset);
    if (offset > std::numeric_limits<std::uint64_t>::max() / kBitsPerByte) {
        return false;
    }
    std::uint64_t begin = offset * kBitsPerByte;
    const std::uint64_t storage_width =
        static_cast<std::uint64_t>(field.size) * kBitsPerByte;
    std::uint64_t width = storage_width;
    if (field.is_bitfield) {
        width = field.bit_size != 0 ? field.bit_size : storage_width;
        if (field.bit_offset > storage_width ||
            width == 0 || width > storage_width - field.bit_offset ||
            begin > std::numeric_limits<std::uint64_t>::max() - field.bit_offset) {
            return false;
        }
        begin += field.bit_offset;
    }
    if (begin > std::numeric_limits<std::uint64_t>::max() - width) {
        return false;
    }
    coverage.push_back({begin, begin + width});
    return true;
}

[[nodiscard]] bool generated_update_preserves_coverage(
    const tinfo_t& existing_type,
    const qstring& name,
    const SynthStruct& replacement) {
    if (!is_generated_name(name, &replacement.naming)) {
        return true;
    }

    std::uint32_t existing_alignment = 0;
    const size_t existing_size = existing_type.get_size(&existing_alignment);
    if (existing_size == BADSIZE || existing_alignment == 0) {
        msg("Structor: Refusing generated-type update for '%s': existing layout is invalid\n",
            name.c_str());
        return false;
    }
    if (replacement.size < existing_size) {
        msg("Structor: Refusing generated-type update for '%s': size would shrink "
            "from %zu to %u bytes without explicit contradictory evidence\n",
            name.c_str(), existing_size, replacement.size);
        return false;
    }

    udt_type_data_t existing_udt;
    if (!existing_type.get_udt_details(&existing_udt)) {
        msg("Structor: Refusing generated-type update for '%s': existing members "
            "cannot be inspected\n", name.c_str());
        return false;
    }

    std::vector<BitCoverageRange> replacement_coverage;
    replacement_coverage.reserve(replacement.fields.size());
    for (const auto& field : replacement.fields) {
        if (!append_synth_field_coverage(replacement_coverage, field)) {
            msg("Structor: Refusing generated-type update for '%s': replacement "
                "contains an invalid non-padding field range\n", name.c_str());
            return false;
        }
    }

    std::sort(replacement_coverage.begin(), replacement_coverage.end(),
              [](const BitCoverageRange& lhs, const BitCoverageRange& rhs) {
                  if (lhs.begin != rhs.begin) {
                      return lhs.begin < rhs.begin;
                  }
                  return lhs.end < rhs.end;
              });
    std::vector<BitCoverageRange> merged;
    merged.reserve(replacement_coverage.size());
    for (const auto& range : replacement_coverage) {
        if (merged.empty() || range.begin > merged.back().end) {
            merged.push_back(range);
        } else {
            merged.back().end = std::max(merged.back().end, range.end);
        }
    }

    for (const auto& member : existing_udt) {
        if (is_persistence_padding_member(member)) {
            continue;
        }

        std::uint64_t width = member.size;
        if (width == 0 && !member.type.empty()) {
            const size_t member_size = member.type.get_size();
            if (member_size != BADSIZE &&
                member_size <= std::numeric_limits<std::uint64_t>::max() / 8) {
                width = static_cast<std::uint64_t>(member_size) * 8;
            }
        }
        if (width == 0 ||
            member.offset > std::numeric_limits<std::uint64_t>::max() - width) {
            msg("Structor: Refusing generated-type update for '%s': existing member "
                "'%s' has invalid bit bounds\n", name.c_str(), member.name.c_str());
            return false;
        }

        const std::uint64_t end = member.offset + width;
        const bool covered = std::any_of(
            merged.begin(), merged.end(), [&](const BitCoverageRange& range) {
                return range.begin <= member.offset && range.end >= end;
            });
        if (!covered) {
            msg("Structor: Refusing generated-type update for '%s': prior member '%s' "
                "at bit range [%llu,%llu) would lose coverage without explicit "
                "contradictory evidence\n",
                name.c_str(), member.name.c_str(),
                static_cast<unsigned long long>(member.offset),
                static_cast<unsigned long long>(end));
            return false;
        }
    }
    return true;
}

[[nodiscard]] qstring canonical_member_type_identity(
    const tinfo_t& type,
    SemanticType semantic,
    std::uint32_t storage_size) {
    qstring identity;
    if (!type.empty()) {
        qtype serialized;
        if (type.serialize(&serialized) && !serialized.empty()) {
            constexpr char kHex[] = "0123456789ABCDEF";
            std::string encoded;
            encoded.reserve(9 + serialized.size() * 2);
            encoded.append("ida-type:");
            for (type_t value : serialized) {
                const auto byte = static_cast<unsigned char>(value);
                encoded.push_back(kHex[byte >> 4]);
                encoded.push_back(kHex[byte & 0x0F]);
            }
            identity = encoded.c_str();
            return identity;
        }

        if (type.print(&identity, nullptr,
                       PRTYPE_1LINE | PRTYPE_TYPE | PRTYPE_NOREGEX) &&
            !identity.empty()) {
            return identity;
        }
    }

    // Unknown/failed-to-print types must not collapse into a concrete member
    // type during interactive reuse.  Keep the storage interpretation and
    // width in the identity so distinct unknown-width evidence remains apart.
    identity.sprnt("<unknown-semantic-%u-size-%u>",
                   unsigned(semantic), storage_size);
    return identity;
}

[[nodiscard]] SemanticType semantic_identity_from_type(const tinfo_t& type) {
    if (type.empty()) {
        return SemanticType::Unknown;
    }
    if (type.is_funcptr()) {
        return SemanticType::FunctionPointer;
    }
    if (type.is_ptr()) {
        tinfo_t pointed = type.get_pointed_object();
        if (!pointed.empty() && (pointed.is_func() || pointed.is_funcptr())) {
            return SemanticType::FunctionPointer;
        }
        return SemanticType::Pointer;
    }
    if (type.is_array()) {
        return SemanticType::Array;
    }
    if (type.is_struct()) {
        return SemanticType::NestedStruct;
    }
    if (type.is_floating()) {
        return type.get_size() == 4 ? SemanticType::Float : SemanticType::Double;
    }
    if (type.is_unsigned()) {
        return SemanticType::UnsignedInteger;
    }
    if (type.is_integral()) {
        return SemanticType::Integer;
    }
    return SemanticType::Unknown;
}

[[nodiscard]] qstring compose_union_type_identity(
    std::vector<std::string> alternatives) {
    std::sort(alternatives.begin(), alternatives.end());
    std::string identity("union{");
    for (const auto& alternative : alternatives) {
        identity.append(std::to_string(alternative.size()));
        identity.push_back(':');
        identity.append(alternative);
        identity.push_back(';');
    }
    identity.push_back('}');
    return qstring(identity.c_str());
}

[[nodiscard]] qstring canonical_union_type_identity(
    const qvector<SynthField::UnionMember>& members) {
    std::vector<std::string> alternatives;
    alternatives.reserve(members.size());
    for (const auto& member : members) {
        const SemanticType semantic = semantic_identity_from_type(member.type);
        const qstring concrete = canonical_member_type_identity(
            member.type, semantic, member.size);
        std::string identity = std::to_string(static_cast<std::int64_t>(member.offset));
        identity.push_back('@');
        identity.append(std::to_string(member.size));
        identity.push_back('@');
        identity.append(std::to_string(static_cast<unsigned>(semantic)));
        identity.push_back('@');
        identity.append(concrete.c_str());
        alternatives.push_back(std::move(identity));
    }
    return compose_union_type_identity(std::move(alternatives));
}

[[nodiscard]] qstring canonical_union_type_identity(const tinfo_t& union_type) {
    udt_type_data_t members;
    if (!union_type.is_union() || !union_type.get_udt_details(&members)) {
        const size_t type_size = union_type.get_size();
        const std::uint32_t storage_size =
            type_size == BADSIZE ||
            type_size > std::numeric_limits<std::uint32_t>::max()
                ? 0
                : static_cast<std::uint32_t>(type_size);
        return canonical_member_type_identity(
            union_type, SemanticType::Unknown, storage_size);
    }

    std::vector<std::string> alternatives;
    alternatives.reserve(members.size());
    for (const auto& member : members) {
        const SemanticType semantic = semantic_identity_from_type(member.type);
        const std::uint32_t storage_size = member.size >
                static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()) * 8
            ? 0
            : static_cast<std::uint32_t>((member.size + 7) / 8);
        const qstring concrete = canonical_member_type_identity(
            member.type, semantic, storage_size);
        std::string identity = std::to_string(member.offset);
        identity.push_back('@');
        identity.append(std::to_string(member.size));
        identity.push_back('@');
        identity.append(std::to_string(static_cast<unsigned>(semantic)));
        identity.push_back('@');
        identity.append(concrete.c_str());
        alternatives.push_back(std::move(identity));
    }
    return compose_union_type_identity(std::move(alternatives));
}

[[nodiscard]] std::optional<std::uint8_t> declared_alignment_code(
    std::uint32_t alignment) noexcept {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        return std::nullopt;
    }

    std::uint8_t code = 1;
    while (alignment > 1) {
        alignment >>= 1;
        if (code == std::numeric_limits<std::uint8_t>::max()) {
            return std::nullopt;
        }
        ++code;
    }
    return code;
}

[[nodiscard]] std::uint32_t packed_member_alignment(
    const udt_type_data_t& udt, std::optional<std::uint32_t> packing) noexcept {
    std::uint32_t result = 1;
    for (const auto& member : udt) {
        if (member.type.empty()) {
            continue;
        }

        std::uint32_t alignment = member.type.get_alignment();
        if (alignment == 0) {
            continue;
        }
        if (packing.has_value()) {
            alignment = std::min(alignment, *packing);
        }
        result = std::max(result, alignment);
    }
    return result;
}

[[nodiscard]] bool configure_synth_udt_layout(
    udt_type_data_t& udt,
    const SynthStruct& synth_struct,
    const qstring& name) {
    const auto pack_code =
        persistence_invariants::ida_udt_pack_code(synth_struct.packing);
    if (!pack_code.has_value()) {
        msg("Structor: Refusing to persist '%s': unsupported packing value %u bytes\n",
            name.c_str(), synth_struct.packing.value_or(0));
        return false;
    }
    if (synth_struct.alignment == 0) {
        msg("Structor: Refusing to persist '%s': effective alignment is zero\n",
            name.c_str());
        return false;
    }

    // udt_type_data_t::pack uses IDA's logarithmic representation: 0 denotes
    // the ABI default and 1..5 denote pack(1)..pack(16).  It must be present
    // while create_udt() computes the layout; applying it afterwards changes
    // metadata without reliably recomputing size/member layout.
    udt.pack = static_cast<uchar>(*pack_code);

    const std::uint32_t member_alignment =
        packed_member_alignment(udt, synth_struct.packing);
    if (synth_struct.alignment < member_alignment) {
        msg("Structor: Refusing to persist '%s': effective alignment %u is below "
            "the packed member requirement %u\n",
            name.c_str(), synth_struct.alignment, member_alignment);
        return false;
    }

    // Effective alignment normally follows from member types and packing.  A
    // larger recovered alignment requires a declared UDT alignment to be
    // representable in IDA.  Do not manufacture a declaration when the normal
    // layout already produces the requested effective alignment.
    if (synth_struct.alignment > member_alignment) {
        const auto sda = declared_alignment_code(synth_struct.alignment);
        if (!sda.has_value()) {
            msg("Structor: Refusing to persist '%s': unsupported effective alignment %u\n",
                name.c_str(), synth_struct.alignment);
            return false;
        }
        udt.sda = static_cast<uchar>(*sda);
    } else {
        udt.sda = 0;
    }
    return true;
}

[[nodiscard]] UdtLayoutExpectation capture_layout_expectation(
    const udt_type_data_t& udt,
    std::uint8_t pack_code,
    std::uint32_t effective_alignment,
    size_t size) {
    UdtLayoutExpectation expected;
    expected.pack_code = pack_code;
    expected.effective_alignment = effective_alignment;
    expected.size = size;
    expected.is_union = udt.is_union;
    expected.members.reserve(udt.size());
    for (const auto& member : udt) {
        ExpectedUdtMember expected_member;
        expected_member.name = member.name;
        expected_member.bit_offset = member.offset;
        expected_member.bit_size = member.size;
        expected_member.is_bitfield = member.is_bitfield();
        if (expected_member.is_bitfield) {
            bitfield_type_data_t details;
            expected_member.bitfield_encoding_valid =
                member.type.get_bitfield_details(&details);
            if (expected_member.bitfield_encoding_valid) {
                expected_member.bitfield_storage_bytes = details.nbytes;
                expected_member.bitfield_width = details.width;
                expected_member.bitfield_is_unsigned = details.is_unsigned;
            }
        }
        expected.members.push_back(std::move(expected_member));
    }
    return expected;
}

[[nodiscard]] bool verify_named_udt_layout(
    const qstring& name,
    const UdtLayoutExpectation& expected,
    tid_t* verified_tid = nullptr) {
    const tid_t tid = get_named_type_tid(name.c_str());
    if (tid == BADADDR) {
        msg("Structor: Round-trip verification failed for '%s': named type is absent\n",
            name.c_str());
        return false;
    }

    tinfo_t named_type;
    if (!named_type.get_type_by_tid(tid)) {
        msg("Structor: Round-trip verification failed for '%s': TID cannot be loaded\n",
            name.c_str());
        return false;
    }

    std::uint32_t actual_alignment = 0;
    const size_t actual_size = named_type.get_size(&actual_alignment);
    udt_type_data_t actual_udt;
    if (actual_size == BADSIZE || !named_type.get_udt_details(&actual_udt)) {
        msg("Structor: Round-trip verification failed for '%s': invalid UDT layout\n",
            name.c_str());
        return false;
    }

    if (actual_udt.is_union != expected.is_union ||
        actual_udt.pack != expected.pack_code ||
        actual_alignment != expected.effective_alignment ||
        actual_udt.effalign != expected.effective_alignment ||
        actual_size != expected.size ||
        actual_udt.total_size != expected.size) {
        msg("Structor: Round-trip verification failed for '%s': "
            "pack=%u/%u align=%u/%u size=%zu/%zu union=%u/%u\n",
            name.c_str(), unsigned(actual_udt.pack), unsigned(expected.pack_code),
            actual_alignment, expected.effective_alignment,
            actual_size, expected.size,
            unsigned(actual_udt.is_union), unsigned(expected.is_union));
        for (const auto& member : actual_udt) {
            msg("Structor:   Persisted member '%s' bit_offset=%llu bit_size=%llu\n",
                member.name.c_str(),
                static_cast<unsigned long long>(member.offset),
                static_cast<unsigned long long>(member.size));
        }
        return false;
    }

    if (actual_udt.size() != expected.members.size()) {
        msg("Structor: Round-trip verification failed for '%s': member count=%zu/%zu\n",
            name.c_str(), actual_udt.size(), expected.members.size());
        return false;
    }

    std::vector<bool> matched(actual_udt.size(), false);
    for (const auto& expected_member : expected.members) {
        bool found = false;
        for (size_t i = 0; i < actual_udt.size(); ++i) {
            if (matched[i]) {
                continue;
            }
            const auto& actual_member = actual_udt[i];
            if (actual_member.name == expected_member.name &&
                actual_member.offset == expected_member.bit_offset &&
                actual_member.size == expected_member.bit_size &&
                actual_member.is_bitfield() == expected_member.is_bitfield) {
                if (expected_member.is_bitfield) {
                    bitfield_type_data_t details;
                    if (!expected_member.bitfield_encoding_valid ||
                        !actual_member.type.get_bitfield_details(&details) ||
                        details.nbytes != expected_member.bitfield_storage_bytes ||
                        details.width != expected_member.bitfield_width ||
                        details.is_unsigned != expected_member.bitfield_is_unsigned) {
                        continue;
                    }
                }
                matched[i] = true;
                found = true;
                break;
            }
        }
        if (!found) {
            msg("Structor: Round-trip verification failed for '%s': member '%s' "
                "is not at bit range [%llu,%llu) with bitfield=%u\n",
                name.c_str(), expected_member.name.c_str(),
                static_cast<unsigned long long>(expected_member.bit_offset),
                static_cast<unsigned long long>(expected_member.bit_offset +
                                                expected_member.bit_size),
                unsigned(expected_member.is_bitfield));
            for (const auto& member : actual_udt) {
                msg("Structor:   Persisted member '%s' bit_offset=%llu bit_size=%llu\n",
                    member.name.c_str(),
                    static_cast<unsigned long long>(member.offset),
                    static_cast<unsigned long long>(member.size));
            }
            return false;
        }
    }

    if (verified_tid != nullptr) {
        *verified_tid = tid;
    }
    return true;
}

[[nodiscard]] std::uint8_t derive_anonymous_udt_pack(
    const udt_type_data_t& udt) noexcept {
    if (udt.pack <= 5) {
        if (udt.pack != 0) {
            return udt.pack;
        }
    }

    // A default-packed anonymous UDT needs an explicit pack only when its
    // observed byte offsets cannot be produced under natural member alignment.
    // Select the largest supported cap that satisfies every informative member;
    // layouts compatible with pack(16)/the ABI default remain default-packed.
    constexpr std::array<std::uint32_t, 5> caps = {16, 8, 4, 2, 1};
    for (std::uint32_t cap : caps) {
        bool compatible = true;
        bool cap_is_material = false;
        for (const auto& member : udt) {
            if (member.type.empty() || (member.offset % 8) != 0) {
                continue;
            }
            const std::uint32_t natural = member.type.get_alignment();
            if (natural == 0) {
                continue;
            }
            const std::uint32_t required = std::min(natural, cap);
            const std::uint64_t byte_offset = member.offset / 8;
            if ((byte_offset % required) != 0) {
                compatible = false;
                break;
            }
            if (natural > cap && (byte_offset % natural) != 0) {
                cap_is_material = true;
            }
        }
        if (compatible) {
            if (!cap_is_material || cap == 16) {
                return 0;
            }
            return *persistence_invariants::ida_udt_pack_code(
                std::optional<std::uint32_t>{cap});
        }
    }
    return 0;
}

void log_array_element_udt(const qstring& field_name, const tinfo_t& type) {
    array_type_data_t atd;
    if (!type.is_array() || !type.get_array_details(&atd)) {
        return;
    }

    if (!atd.elem_type.is_struct() && !atd.elem_type.is_union()) {
        return;
    }

    qstring elem_name;
    atd.elem_type.get_type_name(&elem_name);
    msg("Structor:   Array field '%s' elem_type='%s' nelems=%u elem_size=%zu total_size=%zu\n",
        field_name.c_str(), elem_name.c_str(), unsigned(atd.nelems),
        atd.elem_type.get_size(), type.get_size());

    udt_type_data_t udt;
    if (!atd.elem_type.get_udt_details(&udt)) {
        return;
    }

    for (const auto& member : udt) {
        qstring member_type;
        member.type.print(&member_type);
        msg("Structor:     elem member '%s' off=0x%llX size_bits=%llu type=%s\n",
            member.name.c_str(),
            static_cast<unsigned long long>(member.offset / 8),
            static_cast<unsigned long long>(member.size),
            member_type.c_str());
    }
}

bool extract_function_type_for_reuse(const tinfo_t& type, tinfo_t& out_func_type) {
    if (type.empty()) {
        return false;
    }

    if (type.is_func()) {
        out_func_type = type;
        return true;
    }

    if (type.is_funcptr()) {
        tinfo_t pointed = type.get_pointed_object();
        if (!pointed.empty() && pointed.is_func()) {
            out_func_type = pointed;
            return true;
        }
    }

    if (type.is_ptr()) {
        tinfo_t pointed = type.get_pointed_object();
        if (!pointed.empty() && pointed.is_func()) {
            out_func_type = pointed;
            return true;
        }
    }

    return false;
}

const udm_t* find_member_at_offset(const udt_type_data_t& udt, sval_t offset) {
    for (const auto& member : udt) {
        if (member.offset == static_cast<uint64>(offset) * 8) {
            return &member;
        }
    }

    return nullptr;
}

bool synth_has_nontrivial_unions(const SynthStruct& synth_struct) {
    for (const auto& field : synth_struct.fields) {
        if (field.is_union_candidate && !field.union_members.empty()) {
            return true;
        }
    }

    return false;
}

bool union_has_relative_members(const qvector<SynthField>& members) {
    for (const auto& member : members) {
        if (member.offset != 0) {
            return true;
        }
    }

    return false;
}

bool semantic_blocks_value_enum(SemanticType semantic) {
    switch (semantic) {
        case SemanticType::Pointer:
        case SemanticType::FunctionPointer:
        case SemanticType::VTablePointer:
        case SemanticType::NestedStruct:
        case SemanticType::Float:
        case SemanticType::Double:
        case SemanticType::Array:
        case SemanticType::Padding:
            return true;
        default:
            return false;
    }
}

bool type_blocks_value_enum(const tinfo_t& type) {
    if (type.empty()) {
        return false;
    }

    return type.is_ptr() ||
           type.is_func() ||
           type.is_funcptr() ||
           type.is_array() ||
           type.is_struct() ||
           type.is_union() ||
           type.is_floating();
}

bool field_supports_value_enum(const SynthField& field) {
    if (field.is_bitfield || field.is_padding || field.is_array ||
        field.is_union_candidate || field.size == 0 || field.size > 8) {
        return false;
    }

    if (semantic_blocks_value_enum(field.semantic) ||
        type_blocks_value_enum(field.type)) {
        return false;
    }

    for (const auto& access : field.source_accesses) {
        if (semantic_blocks_value_enum(access.semantic_type) ||
            type_blocks_value_enum(access.inferred_type)) {
            return false;
        }
    }

    return true;
}

std::uint64_t max_storable_value(std::uint32_t size) {
    if (size >= sizeof(std::uint64_t)) {
        return std::numeric_limits<std::uint64_t>::max();
    }

    return (std::uint64_t{1} << (size * 8)) - 1;
}

tid_t create_substruct_recursive(StructurePersistence& persistence, SubStructInfo& sub) {
    if (!sub.children.empty()) {
        return persistence.create_struct_with_substructs(sub.structure, sub.children);
    }
    return persistence.create_struct(sub.structure);
}

tinfo_t make_member_storage_type(const SynthField& member) {
    if (!member.type.empty()) {
        const size_t type_size = member.type.get_size();
        if (type_size != BADSIZE && type_size == member.size) {
            return member.type;
        }

        tinfo_t sized_type;
        switch (member.semantic) {
            case SemanticType::Integer:
            case SemanticType::UnsignedInteger:
            case SemanticType::Float:
            case SemanticType::Double:
                sized_type = utils::create_basic_type(member.size, member.semantic);
                break;
            case SemanticType::Pointer:
            case SemanticType::FunctionPointer:
            case SemanticType::VTablePointer:
                if (member.size == get_ptr_size()) {
                    sized_type = utils::create_basic_type(member.size, member.semantic);
                }
                break;
            default:
                break;
        }

        const size_t sized_type_size = sized_type.get_size();
        if (!sized_type.empty() && sized_type_size != BADSIZE &&
            sized_type_size == member.size) {
            msg("Structor:   Normalizing field '%s' type width from %zu to %u bytes\n",
                member.name.c_str(), type_size, member.size);
            return sized_type;
        }
    }

    tinfo_t byte_type;
    byte_type.create_simple_type(BT_INT8 | BTMT_CHAR);

    tinfo_t storage_type;
    if (member.size > 1) {
        if (!storage_type.create_array(byte_type, member.size)) {
            return tinfo_t();
        }
    } else if (member.size == 1) {
        storage_type = byte_type;
    }

    return storage_type;
}

tinfo_t make_bitfield_storage_type(const SynthField& field) {
    if (field.size != 1 && field.size != 2 &&
        field.size != 4 && field.size != 8) {
        return tinfo_t();
    }

    const std::uint32_t width = field.bit_size != 0
        ? field.bit_size
        : field.size * 8;
    if (width == 0 || width > field.size * 8 ||
        width > std::numeric_limits<uchar>::max()) {
        return tinfo_t();
    }

    const bool is_unsigned =
        field.semantic == SemanticType::UnsignedInteger ||
        (!field.type.empty() && field.type.is_unsigned());
    tinfo_t type;
    if (!type.create_bitfield(static_cast<uchar>(field.size),
                              static_cast<uchar>(width),
                              is_unsigned)) {
        return tinfo_t();
    }
    return type;
}

bool append_bitfield_tail_padding(udt_type_data_t& udt,
                                  const qvector<SynthField>& fields) {
    std::vector<std::pair<sval_t, std::uint32_t>> groups;
    for (const auto& field : fields) {
        if (!field.is_bitfield) {
            continue;
        }
        const auto key = std::make_pair(field.offset, field.size);
        if (std::find(groups.begin(), groups.end(), key) == groups.end()) {
            groups.push_back(key);
        }
    }

    for (const auto& [offset, storage_size] : groups) {
        if (offset < 0 || (storage_size != 1 && storage_size != 2 &&
                           storage_size != 4 && storage_size != 8)) {
            return false;
        }

        const std::uint32_t storage_bits = storage_size * 8;
        std::uint32_t used_end = 0;
        for (const auto& field : fields) {
            if (!field.is_bitfield || field.offset != offset ||
                field.size != storage_size) {
                continue;
            }
            const std::uint32_t width = field.bit_size != 0
                ? field.bit_size
                : storage_bits;
            if (field.bit_offset > storage_bits ||
                width > storage_bits - field.bit_offset) {
                return false;
            }
            used_end = std::max(used_end, field.bit_offset + width);
        }

        if (used_end >= storage_bits) {
            continue;
        }

        const std::uint32_t remaining = storage_bits - used_end;
        udm_t tail;
        tail.name.sprnt("__pad_bits_%s_%u",
                        make_offset_suffix(offset).c_str(), used_end);
        tail.offset = static_cast<std::uint64_t>(offset) * 8 + used_end;
        tail.size = remaining;
        if (!tail.type.create_bitfield(static_cast<uchar>(storage_size),
                                       static_cast<uchar>(remaining), true)) {
            return false;
        }
        udt.push_back(std::move(tail));
    }

    std::sort(udt.begin(), udt.end(), [](const udm_t& lhs, const udm_t& rhs) {
        if (lhs.offset != rhs.offset) {
            return lhs.offset < rhs.offset;
        }
        return lhs.name < rhs.name;
    });
    return true;
}

} // namespace

tinfo_t StructurePersistence::create_overlay_view_type(
    const qstring& union_name,
    const SynthField& member,
    uint32_t union_size) {
    qstring type_name = make_internal_overlay_view_type_name(union_name,
                                                             member.name,
                                                             member.offset);

    udt_type_data_t udt;
    udt.is_union = false;
    udt.total_size = union_size;
    udt.pack = 1;  // Intentional byte-window view: IDA pack(1).
    udt.sda = 1;   // Intentional declared/effective alignment of one byte.

    udm_t udm;
    udm.name = member.name;
    udm.offset = static_cast<uint64>(member.offset) * 8;
    udm.type = make_member_storage_type(member);
    const size_t type_size = udm.type.get_size();
    udm.size = type_size != BADSIZE ? type_size * 8 : member.size * 8;
    udt.push_back(udm);

    const UdtLayoutExpectation expected =
        capture_layout_expectation(udt, 1, 1, union_size);

    tinfo_t existing;
    tid_t existing_tid = get_named_type_tid(type_name.c_str());
    if (existing_tid != BADADDR) {
        if (is_structor_owned_type(existing_tid) &&
            verify_named_udt_layout(type_name, expected, &existing_tid) &&
            existing.get_type_by_tid(existing_tid)) {
            return existing;
        }
        msg("Structor: Refusing to replace incompatible overlay helper '%s'\n",
            type_name.c_str());
        return tinfo_t();
    }

    tinfo_t view_type;
    if (!view_type.create_udt(udt)) {
        return tinfo_t();
    }

    if (set_named_type_transactional(view_type, type_name) != TERR_OK) {
        return tinfo_t();
    }

    tinfo_t named_type;
    tid_t tid = BADADDR;
    if (verify_named_udt_layout(type_name, expected, &tid) &&
        named_type.get_type_by_tid(tid)) {
        if (store_provenance(tid, {})) {
            return named_type;
        }
        poison_transaction();
        return tinfo_t();
    }

    poison_transaction();
    return tinfo_t();
}

namespace {

tinfo_t create_overlay_array_type(const SynthField& member, uint32_t union_size) {
    if (member.size == 0 || member.offset < 0 || union_size == 0 || (union_size % member.size) != 0) {
        return tinfo_t();
    }

    tinfo_t element_type = make_member_storage_type(member);
    const size_t elem_size = element_type.get_size();
    if (elem_size == BADSIZE || elem_size != member.size) {
        return tinfo_t();
    }

    tinfo_t array_type;
    if (!array_type.create_array(element_type, union_size / member.size)) {
        return tinfo_t();
    }

    return array_type;
}

bool reuse_candidate_matches_function_members(const SynthStruct& synth_struct, const tinfo_t& tif) {
    udt_type_data_t udt;
    if (!tif.get_udt_details(&udt)) {
        return false;
    }

    for (const auto& field : synth_struct.fields) {
        if (field.type.empty()) {
            continue;
        }

        tinfo_t observed_func;
        if (!extract_function_type_for_reuse(field.type, observed_func)) {
            continue;
        }

        const udm_t* member = find_member_at_offset(udt, field.offset);
        if (!member || member->type.empty()) {
            return false;
        }

        tinfo_t candidate_func;
        if (!extract_function_type_for_reuse(member->type, candidate_func)) {
            return false;
        }

        if (!candidate_func.compare_with(observed_func, TCMP_CALL | TCMP_IGNMODS)) {
            return false;
        }
    }

    return true;
}

bool is_functionish_type(const tinfo_t& type) {
    if (type.empty()) {
        return false;
    }
    if (type.is_func() || type.is_funcptr()) {
        return true;
    }
    if (!type.is_ptr()) {
        return false;
    }
    tinfo_t pointed = type.get_pointed_object();
    return !pointed.empty() && pointed.is_func();
}

bool field_has_vtable_evidence(const SynthField& field, sval_t parent_offset) {
    if (field.offset != parent_offset || field.size != get_ptr_size()) {
        return false;
    }

    if (field.semantic == SemanticType::VTablePointer) {
        return true;
    }

    for (const auto& access : field.source_accesses) {
        if (access.is_vtable_access || access.semantic_type == SemanticType::VTablePointer) {
            return true;
        }
    }

    return false;
}

[[nodiscard]] bool apply_vtable_pointer_type(SynthStruct& synth_struct) {
    if (!synth_struct.has_vtable() || synth_struct.vtable->tid == BADADDR) {
        return false;
    }

    tinfo_t vtbl_type;
    if (!vtbl_type.get_type_by_tid(synth_struct.vtable->tid)) {
        return false;
    }

    tinfo_t vtbl_ptr_type;
    if (!vtbl_ptr_type.create_ptr(vtbl_type)) {
        return false;
    }

    for (auto& field : synth_struct.fields) {
        if (!field_has_vtable_evidence(field, synth_struct.vtable->parent_offset)) {
            continue;
        }

        field.semantic = SemanticType::VTablePointer;
        field.type = vtbl_ptr_type;
        // A raw pointer-sized load can coexist with indirect-call evidence at
        // the same offset. Once the latter has materialized a concrete vtable,
        // it is the authoritative interpretation; retaining the preliminary
        // integer/pointer alternatives would persist a stale union instead of
        // the vtable pointer required by the parent type.
        field.is_union_candidate = false;
        field.union_members.clear();

        if (field.name.empty() || is_generated_name(field.name, &field.naming)) {
            qstring preferred_name;
            if (field.offset == 0) {
                preferred_name = "vtable";
            } else {
                preferred_name.sprnt("vtable_%s", make_offset_suffix(field.offset).c_str());
            }
            set_generated_name(field.name,
                               field.naming,
                               preferred_name,
                               GeneratedNameKind::Field,
                               NameConfidence::High);
        }
        return true;
    }
    return false;
}

void apply_array_element_role_names(udt_type_data_t& udt) {
    int func_members = 0;
    for (const auto& member : udt) {
        if (is_functionish_type(member.type)) {
            ++func_members;
        }
    }

    if (func_members == 0) {
        return;
    }

    for (auto& member : udt) {
        if (!is_functionish_type(member.type) || !is_generated_name(member.name)) {
            continue;
        }

        if (func_members == 1) {
            member.name = "callback";
        } else {
            member.name.sprnt("callback_%llX",
                              static_cast<unsigned long long>(member.offset / 8));
        }
    }
}

} // namespace

StructurePersistence::Transaction::Transaction(Transaction&& other) noexcept
    : owner_(std::move(other.owner_))
    , unresolved_(std::exchange(other.unresolved_, false)) {
    other.owner_.reset();
}

StructurePersistence::Transaction& StructurePersistence::Transaction::operator=(
    Transaction&& other) noexcept {
    if (this != &other) {
        if (unresolved_) {
            if (auto token = owner_.lock(); token && token->owner != nullptr &&
                !token->owner->rollback_transaction()) {
                msg("Structor: CRITICAL: persistence transaction rollback failed "
                    "during move\n");
            }
        }
        owner_ = std::move(other.owner_);
        unresolved_ = std::exchange(other.unresolved_, false);
        other.owner_.reset();
    }
    return *this;
}

StructurePersistence::Transaction::~Transaction() noexcept {
    if (unresolved_) {
        if (auto token = owner_.lock(); token && token->owner != nullptr &&
            !token->owner->rollback_transaction()) {
            msg("Structor: CRITICAL: persistence transaction rollback failed\n");
        }
    }
}

bool StructurePersistence::Transaction::commit() noexcept {
    if (!unresolved_) {
        return true;
    }
    unresolved_ = false;
    auto token = owner_.lock();
    owner_.reset();
    if (!token || token->owner == nullptr) {
        return false;
    }
    return token->owner->commit_transaction();
}

bool StructurePersistence::Transaction::rollback() noexcept {
    if (!unresolved_) {
        return true;
    }
    unresolved_ = false;
    auto token = owner_.lock();
    owner_.reset();
    if (!token || token->owner == nullptr) {
        return true;
    }
    return token->owner->rollback_transaction();
}

bool StructurePersistence::Transaction::active() const noexcept {
    if (!unresolved_) {
        return false;
    }
    const auto token = owner_.lock();
    return token && token->owner != nullptr &&
        token->owner->transaction_active();
}

StructurePersistence::~StructurePersistence() noexcept {
    if (transaction_state_ != nullptr && !rollback_transaction()) {
        msg("Structor: CRITICAL: persistence transaction rollback failed "
            "during owner destruction\n");
    }
    if (transaction_owner_token_) {
        transaction_owner_token_->owner = nullptr;
    }
}

std::optional<StructurePersistence::Transaction>
StructurePersistence::begin_transaction() {
    if (transaction_state_ != nullptr) {
        msg("Structor: Refusing nested persistence transaction\n");
        return std::nullopt;
    }
    transaction_state_ = std::make_unique<TransactionState>();
    return Transaction(transaction_owner_token_);
}

bool StructurePersistence::transaction_active() const noexcept {
    return transaction_state_ != nullptr;
}

bool StructurePersistence::transaction_poisoned() const noexcept {
    return transaction_state_ != nullptr && transaction_state_->poisoned;
}

void StructurePersistence::poison_transaction() noexcept {
    if (transaction_state_ != nullptr && !transaction_state_->rolling_back) {
        transaction_state_->poisoned = true;
    }
}

bool StructurePersistence::stage_auxiliary_named_type(
    const qstring& name,
    const tinfo_t& definition,
    tinfo_t& out_named_type) {
    if (!transaction_active() || name.empty() || definition.empty()) {
        return false;
    }

    const tid_t existing_tid = get_named_type_tid(name.c_str());
    if (existing_tid != BADADDR) {
        tinfo_t existing;
        if (!existing.get_type_by_tid(existing_tid)) {
            return false;
        }
        if (existing.equals_to(definition)) {
            out_named_type = existing;
            return true;
        }
        if (!is_structor_owned_type(existing_tid)) {
            return false;
        }
    }

    tinfo_t staged = definition;
    const int flags = NTF_TYPE |
        (existing_tid == BADADDR ? 0 : NTF_REPLACE);
    if (set_named_type_transactional(staged, name, flags) != TERR_OK) {
        return false;
    }

    const tid_t staged_tid = get_named_type_tid(name.c_str());
    tinfo_t observed;
    if (staged_tid == BADADDR || !observed.get_type_by_tid(staged_tid) ||
        !observed.equals_to(definition)) {
        poison_transaction();
        return false;
    }
    if (!store_provenance(staged_tid, {})) {
        poison_transaction();
        return false;
    }
    out_named_type = observed;
    return true;
}

StructurePersistence::PreparedNamedTypeWrite
StructurePersistence::prepare_named_type_write(const qstring& name) {
    PreparedNamedTypeWrite prepared;
    if (transaction_state_ == nullptr || transaction_state_->rolling_back) {
        return prepared;
    }
    if (name.empty()) {
        prepared.allowed = false;
        return prepared;
    }

    const std::string key(name.c_str());
    if (const auto existing = transaction_state_->journal.find(key)) {
        prepared.journal_index = *existing;
        return prepared;
    }

    const tid_t prior_tid = get_named_type_tid(name.c_str());
    const bool existed_before = prior_tid != BADADDR;
    NamedTypeSnapshot snapshot;
    if (existed_before) {
        if (!is_structor_owned_type(prior_tid)) {
            msg("Structor: Refusing transactional replacement of unowned type '%s'\n",
                name.c_str());
            prepared.allowed = false;
            return prepared;
        }
        if (!snapshot.type.get_type_by_tid(prior_tid) ||
            !snapshot.type.detach()) {
            msg("Structor: Cannot snapshot owned type '%s' before replacement\n",
                name.c_str());
            prepared.allowed = false;
            return prepared;
        }
        snapshot.valid = true;
        snapshot.original_tid = prior_tid;
        snapshot.provenance = load_provenance(prior_tid);
    }

    try {
        prepared.journal_index = transaction_state_->journal.stage(
            key, existed_before,
            existed_before ? static_cast<std::uint64_t>(prior_tid) : 0);
        if (!prepared.journal_index.has_value()) {
            prepared.allowed = false;
            poison_transaction();
            return prepared;
        }
        if (transaction_state_->snapshots.size() <= *prepared.journal_index) {
            transaction_state_->snapshots.resize(*prepared.journal_index + 1);
        }
        transaction_state_->snapshots[*prepared.journal_index] =
            std::move(snapshot);
    } catch (...) {
        poison_transaction();
        prepared.allowed = false;
        prepared.journal_index.reset();
    }
    return prepared;
}

void StructurePersistence::mark_named_type_written(
    const PreparedNamedTypeWrite& prepared,
    tid_t current_tid) {
    if (transaction_state_ == nullptr || transaction_state_->rolling_back ||
        !prepared.journal_index.has_value()) {
        return;
    }

    if (current_tid == BADADDR) {
        // A failed replacement may remove the old name entirely. If this is
        // the first observed mutation, retain the original identity solely as
        // a nonzero rollback token; rollback accepts a missing current name.
        const auto* entry = transaction_state_->journal.entry(
            *prepared.journal_index);
        if (entry == nullptr || entry->mutated || !entry->existed_before ||
            entry->original_identity == 0) {
            return;
        }
        (void)transaction_state_->journal.mark_mutated(
            *prepared.journal_index, entry->original_identity);
        return;
    }

    (void)transaction_state_->journal.mark_mutated(
        *prepared.journal_index, static_cast<std::uint64_t>(current_tid));
}

tinfo_code_t StructurePersistence::set_named_type_transactional(
    tinfo_t& type,
    const qstring& name,
    int ntf_flags) {
    const PreparedNamedTypeWrite prepared = prepare_named_type_write(name);
    if (!prepared.allowed) {
        return TERR_SAVE_ERROR;
    }
    tinfo_code_t result = TERR_SAVE_ERROR;
    try {
        result = type.set_named_type(nullptr, name.c_str(), ntf_flags);
    } catch (...) {
        try {
            mark_named_type_written(
                prepared, get_named_type_tid(name.c_str()));
        } catch (...) {
            mark_named_type_written(prepared, BADADDR);
        }
        poison_transaction();
        return TERR_SAVE_ERROR;
    }
    if (result != TERR_OK) {
        // IDA can report a save error after partially changing the numbered
        // type. Mark any surviving name as mutated so the first-touch snapshot
        // remains authoritative during rollback.
        const tid_t partial_tid = get_named_type_tid(name.c_str());
        mark_named_type_written(prepared, partial_tid);
        poison_transaction();
        return result;
    }

    const tid_t tid = get_named_type_tid(name.c_str());
    if (tid == BADADDR) {
        mark_named_type_written(prepared, BADADDR);
        poison_transaction();
        msg("Structor: Named type '%s' vanished immediately after persistence\n",
            name.c_str());
        return TERR_SAVE_ERROR;
    }
    mark_named_type_written(prepared, tid);
    return TERR_OK;
}

void StructurePersistence::kill_provenance_record(tid_t tid) {
    if (tid == BADADDR) {
        return;
    }
    qstring node_name;
    node_name.sprnt("%s%llX", PROVENANCE_NETNODE_PREFIX,
                    static_cast<unsigned long long>(tid));
    netnode node(node_name.c_str(), 0, false);
    if (node != BADNODE) {
        node.kill();
    }
}

bool StructurePersistence::commit_transaction() noexcept {
    if (transaction_state_ == nullptr) {
        return true;
    }
    if (transaction_state_->poisoned) {
        msg("Structor: Refusing to commit a poisoned persistence transaction; "
            "rolling back\n");
        if (!rollback_transaction()) {
            msg("Structor: CRITICAL: poisoned persistence transaction rollback "
                "failed\n");
        }
        return false;
    }
    transaction_state_.reset();
    return true;
}

bool StructurePersistence::rollback_transaction() noexcept {
    if (transaction_state_ == nullptr) {
        return true;
    }

    bool success = true;
    transaction_state_->rolling_back = true;
    try {
        const auto order = transaction_state_->journal.rollback_order();
        for (const std::size_t index : order) {
            const auto* entry = transaction_state_->journal.entry(index);
            if (entry == nullptr) {
                success = false;
                continue;
            }

            const tid_t current_tid = get_named_type_tid(entry->name.c_str());
            if (current_tid != BADADDR &&
                static_cast<std::uint64_t>(current_tid) != entry->current_identity) {
                msg("Structor: Refusing rollback of '%s': current TID changed\n",
                    entry->name.c_str());
                success = false;
                continue;
            }

            if (!entry->existed_before) {
                if (current_tid == BADADDR) {
                    kill_provenance_record(
                        static_cast<tid_t>(entry->current_identity));
                    continue;
                }
                if (!del_named_type(nullptr, entry->name.c_str(), NTF_TYPE)) {
                    msg("Structor: Failed to remove new transactional type '%s'\n",
                        entry->name.c_str());
                    success = false;
                    continue;
                }
                kill_provenance_record(current_tid);
                msg("Structor: Transaction rollback removed new type '%s'\n",
                    entry->name.c_str());
                continue;
            }

            if (index >= transaction_state_->snapshots.size() ||
                !transaction_state_->snapshots[index].valid) {
                msg("Structor: Missing rollback snapshot for '%s'\n",
                    entry->name.c_str());
                success = false;
                continue;
            }
            const auto& snapshot = transaction_state_->snapshots[index];
            tinfo_t restored_type = snapshot.type;
            if (restored_type.set_named_type(
                    nullptr, entry->name.c_str(), NTF_TYPE | NTF_REPLACE) != TERR_OK) {
                msg("Structor: Failed to restore transactional type '%s'\n",
                    entry->name.c_str());
                success = false;
                continue;
            }

            const tid_t restored_tid = get_named_type_tid(entry->name.c_str());
            tinfo_t verified_type;
            if (restored_tid == BADADDR ||
                !verified_type.get_type_by_tid(restored_tid) ||
                !verified_type.equals_to(snapshot.type)) {
                msg("Structor: Restored transactional type '%s' failed verification\n",
                    entry->name.c_str());
                success = false;
                continue;
            }
            const tid_t transactional_tid = current_tid != BADADDR
                ? current_tid
                : static_cast<tid_t>(entry->current_identity);
            if (transactional_tid != BADADDR &&
                transactional_tid != restored_tid) {
                kill_provenance_record(transactional_tid);
            }
            if (snapshot.original_tid != BADADDR &&
                snapshot.original_tid != restored_tid &&
                snapshot.original_tid != transactional_tid) {
                kill_provenance_record(snapshot.original_tid);
            }
            if (!store_provenance(restored_tid, snapshot.provenance)) {
                success = false;
            }
            const qvector<ea_t> restored_provenance =
                load_provenance(restored_tid);
            if (!is_structor_owned_type(restored_tid) ||
                restored_provenance.size() != snapshot.provenance.size() ||
                !std::equal(restored_provenance.begin(),
                            restored_provenance.end(),
                            snapshot.provenance.begin())) {
                msg("Structor: Restored transactional metadata for '%s' "
                    "failed verification\n", entry->name.c_str());
                success = false;
                continue;
            }
            msg("Structor: Transaction rollback restored type '%s'\n",
                entry->name.c_str());
        }
    } catch (...) {
        success = false;
        msg("Structor: Exception during persistence transaction rollback\n");
    }
    transaction_state_.reset();
    return success;
}

tid_t StructurePersistence::create_struct(SynthStruct& synth_struct) {
    std::optional<Transaction> owned_transaction;
    if (!transaction_active()) {
        owned_transaction = begin_transaction();
        if (!owned_transaction.has_value()) {
            return BADADDR;
        }
    }
    tid_t result = BADADDR;
    try {
        result = create_struct_impl(synth_struct);
    } catch (...) {
        poison_transaction();
    }
    if (result == BADADDR) {
        poison_transaction();
    }
    if (result != BADADDR && transaction_poisoned()) {
        result = BADADDR;
        synth_struct.tid = BADADDR;
    }
    if (owned_transaction.has_value()) {
        if (result != BADADDR) {
            if (!owned_transaction->commit()) {
                result = BADADDR;
                synth_struct.tid = BADADDR;
            }
        } else if (!owned_transaction->rollback()) {
            msg("Structor: CRITICAL: implicit struct transaction rollback failed\n");
        }
    }
    return result;
}

tid_t StructurePersistence::create_struct_impl(SynthStruct& synth_struct) {
    if (!validate_public_struct_shape(synth_struct)) {
        msg("Structor: Refusing invalid public structure shape '%s'\n",
            synth_struct.name.c_str());
        return BADADDR;
    }
    constexpr double kReuseThreshold = 0.85;
    const bool allow_reuse = synth_struct.naming.origin != NameOrigin::HeuristicRole;

    qstring name = synth_struct.name;
    const tid_t named_target_tid = name.empty()
        ? BADADDR
        : get_named_type_tid(name.c_str());
    const bool existing_generated_target =
        named_target_tid != BADADDR &&
        is_generated_name(name, &synth_struct.naming) &&
        is_structor_owned_type(named_target_tid);

    // Decide external-root reuse before creating any companion types. A reused
    // root is not rewritten to reference a newly generated vtable, so creating
    // that vtable first would commit an orphaned auxiliary type.
    if (!existing_generated_target && options_.interactive_mode && allow_reuse &&
        !synth_has_nontrivial_unions(synth_struct) &&
        !synth_struct.fields.empty() && synth_struct.size > 0) {
        auto reuse_candidate = find_reuse_candidate(synth_struct, kReuseThreshold);
        if (reuse_candidate.has_value()) {
            auto [reuse_tid, reuse_name, score] = *reuse_candidate;
            qstring prompt;
            prompt.sprnt("Structor: Reuse existing struct '%s' (%.0f%% match)?",
                         reuse_name.c_str(), score * 100.0);
            const bool reuse =
                ask_yn(ASKBTN_YES, "%s", prompt.c_str()) == ASKBTN_YES;
            if (reuse) {
                synth_struct.name = reuse_name;
                synth_struct.tid = reuse_tid;

                if (reuse_tid != BADADDR && is_structor_owned_type(reuse_tid)) {
                    qvector<ea_t> merged = get_provenance(reuse_tid);
                    for (ea_t ea : synth_struct.provenance) {
                        if (std::find(merged.begin(), merged.end(), ea) == merged.end()) {
                            merged.push_back(ea);
                        }
                    }
                    if (!set_provenance(reuse_tid, merged)) {
                        return BADADDR;
                    }
                }
                return reuse_tid;
            }
        }
    }

    if (synth_struct.has_vtable()) {
        tid_t vtbl_tid = create_vtable(*synth_struct.vtable);
        if (vtbl_tid == BADADDR) {
            msg("Structor: Refusing to persist '%s': vtable creation failed\n",
                name.c_str());
            return BADADDR;
        }
        synth_struct.vtable->tid = vtbl_tid;
        if (!apply_vtable_pointer_type(synth_struct)) {
            msg("Structor: Refusing to persist '%s': vtable pointer field "
                "could not be materialized\n", name.c_str());
            return BADADDR;
        }
    }

    if (existing_generated_target) {
        tid_t existing_tid = named_target_tid;
        if (existing_tid != BADADDR && update_struct(existing_tid, synth_struct)) {
            synth_struct.name = name;
            synth_struct.tid = existing_tid;
            return existing_tid;
        }
        msg("Structor: Failed or refused update of generated type '%s'; "
            "not creating a duplicate replacement\n", name.c_str());
        return BADADDR;
    }

    // Generate unique name if needed
    name = synth_struct.name;
    if (struct_exists(name.c_str())) {
        name = make_unique_name(name.c_str());
        synth_struct.name = name;
    }

    // Create the structure type
    tinfo_t struct_type;
    udt_type_data_t udt;
    udt.is_union = false;
    udt.total_size = synth_struct.size;

    std::unordered_map<uint64_t, tinfo_t> value_enums;
    for (const auto& field : synth_struct.fields) {
        if (field.is_bitfield) {
            continue;
        }

        uint64_t key = (static_cast<uint64_t>(field.offset) << 32) | field.size;
        if (value_enums.find(key) == value_enums.end()) {
            tinfo_t enum_type = create_value_enum_type(name, field);
            if (!enum_type.empty()) {
                value_enums.emplace(key, enum_type);
            }
        }
    }

    // Add fields
    msg("Structor: Creating struct '%s' with %zu fields, total_size=%u\n",
        name.c_str(), synth_struct.fields.size(), synth_struct.size);
    for (const auto& field : synth_struct.fields) {
        if (field.is_union_candidate && !field.union_members.empty()) {
            qvector<SynthField> members;
            members.reserve(field.union_members.size());
            for (const auto& alt : field.union_members) {
                SynthField member;
                member.name = alt.name;
                member.offset = alt.offset;
                member.size = alt.size;
                member.type = alt.type;
                member.comment = alt.comment;
                members.push_back(std::move(member));
            }

            qstring union_name = field.name.empty() ? qstring("union") : field.name;
            if (add_union_field(udt, field.offset, union_name, members) != BADADDR) {
                continue;
            }
        }

        msg("Structor:   Adding field '%s' at offset 0x%llX (bits: 0x%llX), size=%u\n",
            field.name.c_str(), static_cast<unsigned long long>(field.offset),
            static_cast<unsigned long long>(field.offset) * 8, field.size);
        udm_t udm;
        udm.name = field.name;

        if (field.is_bitfield) {
            udm.offset = static_cast<uint64>(field.offset) * 8 + field.bit_offset;
            udm.size = field.bit_size > 0 ? field.bit_size : field.size * 8;
            udm.type = make_bitfield_storage_type(field);
            if (udm.type.empty()) {
                msg("Structor: Refusing invalid bitfield '%s' (storage=%u width=%u)\n",
                    field.name.c_str(), field.size, field.bit_size);
                return BADADDR;
            }
        } else {
            udm.offset = static_cast<uint64>(field.offset) * 8;  // Convert to bits

            uint64_t key = (static_cast<uint64_t>(field.offset) << 32) | field.size;
            auto value_enum_it = value_enums.find(key);

            if (value_enum_it != value_enums.end()) {
                udm.type = value_enum_it->second;
                udm.size = field.size * 8;
            } else if (!field.type.empty()) {
                SynthField storage_field = field;
                storage_field.type = materialize_nested_type(name, field, field.type);
                udm.type = make_member_storage_type(storage_field);
                udm.size = field.size * 8;
            } else {
                // Default to bytes array for unknown types
                tinfo_t byte_type;
                byte_type.create_simple_type(BT_INT8 | BTMT_CHAR);
                if (field.size > 1) {
                    if (!udm.type.create_array(byte_type, field.size)) {
                        msg("Structor: Failed to create storage for field '%s'\n",
                            field.name.c_str());
                        return BADADDR;
                    }
                } else {
                    udm.type = byte_type;
                }
                udm.size = field.size * 8;
            }
        }

        log_array_element_udt(field.name, udm.type);

        if (!field.comment.empty()) {
            udm.cmt = field.comment;
        }

        udt.push_back(udm);
    }

    if (!append_bitfield_tail_padding(udt, synth_struct.fields)) {
        msg("Structor: Refusing '%s': invalid bitfield storage layout\n",
            name.c_str());
        return BADADDR;
    }
    if (!configure_synth_udt_layout(udt, synth_struct, name)) {
        return BADADDR;
    }
    const UdtLayoutExpectation expected = capture_layout_expectation(
        udt, udt.pack, synth_struct.alignment, synth_struct.size);

    // Create the struct type with packing present during layout calculation.
    if (!struct_type.create_udt(udt)) {
        msg("Structor: Failed to create struct type\n");
        return BADADDR;
    }

    if (persistence_invariants::persistence_fault_requested("before_root_write")) {
        msg("Structor: Integration fault injected before root type write\n");
        return BADADDR;
    }

    // Save to local type library
    tinfo_code_t err = set_named_type_transactional(struct_type, name);
    if (err != TERR_OK) {
        msg("Structor: Failed to save struct type: %d\n", err);
        return BADADDR;
    }

    tid_t tid = BADADDR;
    if (!verify_named_udt_layout(name, expected, &tid)) {
        return BADADDR;
    }

    // Store provenance
    if (!store_provenance(tid, synth_struct.provenance)) {
        msg("Structor: Failed to persist verified ownership metadata for '%s'\n",
            name.c_str());
        return BADADDR;
    }
    synth_struct.tid = tid;

    return tid;
}

tid_t StructurePersistence::create_struct_with_substructs(
    SynthStruct& synth_struct,
    qvector<SubStructInfo>& sub_structs)
{
    std::optional<Transaction> owned_transaction;
    if (!transaction_active()) {
        owned_transaction = begin_transaction();
        if (!owned_transaction.has_value()) {
            return BADADDR;
        }
    }
    tid_t result = BADADDR;
    try {
        result = create_struct_with_substructs_impl(synth_struct, sub_structs);
    } catch (...) {
        poison_transaction();
    }
    if (result == BADADDR) {
        poison_transaction();
    }
    if (result != BADADDR && transaction_poisoned()) {
        result = BADADDR;
        synth_struct.tid = BADADDR;
    }
    if (owned_transaction.has_value()) {
        if (result != BADADDR) {
            if (!owned_transaction->commit()) {
                result = BADADDR;
                synth_struct.tid = BADADDR;
            }
        } else if (!owned_transaction->rollback()) {
            msg("Structor: CRITICAL: implicit hierarchy transaction rollback failed\n");
        }
    }
    return result;
}

tid_t StructurePersistence::create_struct_with_substructs_impl(
    SynthStruct& synth_struct,
    qvector<SubStructInfo>& sub_structs)
{
    if (!sub_structs.empty()) {
        for (size_t sub_index = 0; sub_index < sub_structs.size(); ++sub_index) {
            auto& sub = sub_structs[sub_index];
            for (size_t prior = 0; prior < sub_index; ++prior) {
                if (sub_structs[prior].parent_offset == sub.parent_offset) {
                    msg("Structor: Refusing duplicate child offset 0x%llX in parent '%s'\n",
                        static_cast<unsigned long long>(sub.parent_offset),
                        synth_struct.name.c_str());
                    return BADADDR;
                }
            }
            const auto child_end = checked_interval_end(
                sub.parent_offset, sub.structure.size);
            if (sub.parent_offset < 0 || sub.structure.size == 0 ||
                !child_end.has_value() || *child_end < 0 ||
                static_cast<std::uint64_t>(*child_end) > MAX_STRUCT_SIZE) {
                msg("Structor: Refusing child '%s': invalid parent range\n",
                    sub.structure.name.c_str());
                return BADADDR;
            }
            tid_t sub_tid = create_substruct_recursive(*this, sub);
            if (sub_tid == BADADDR) {
                msg("Structor: Refusing parent '%s': child '%s' failed to persist\n",
                    synth_struct.name.c_str(), sub.structure.name.c_str());
                return BADADDR;
            }

            tinfo_t sub_type;
            if (!sub_type.get_type_by_tid(sub_tid)) {
                msg("Structor: Refusing parent '%s': child '%s' TID cannot be loaded\n",
                    synth_struct.name.c_str(), sub.structure.name.c_str());
                return BADADDR;
            }

            bool matched = false;
            SynthField* sole_nested_match = nullptr;
            size_t nested_matches = 0;
            for (auto& field : synth_struct.fields) {
                if (field.offset == sub.parent_offset &&
                    (field.name == sub.field_name || field.name.empty())) {
                    field.type = sub_type;
                    field.semantic = SemanticType::NestedStruct;
                    field.size = sub.structure.size;
                    if (field.name.empty()) {
                        field.name = sub.field_name;
                    }
                    matched = true;
                    break;
                }
                if (field.offset == sub.parent_offset &&
                    field.semantic == SemanticType::NestedStruct &&
                    !field.is_union_candidate) {
                    sole_nested_match = &field;
                    ++nested_matches;
                }
            }

            if (!matched && nested_matches == 1) {
                sole_nested_match->type = sub_type;
                sole_nested_match->semantic = SemanticType::NestedStruct;
                sole_nested_match->size = sub.structure.size;
                if (sole_nested_match->name.empty()) {
                    sole_nested_match->name = sub.field_name;
                } else {
                    sub.field_name = sole_nested_match->name;
                    sub.field_naming = sole_nested_match->naming;
                }
                matched = true;
            }

            if (!matched) {
                for (const auto& field : synth_struct.fields) {
                    const auto field_end = checked_interval_end(
                        field.offset, field.size);
                    if (!field_end.has_value()) {
                        msg("Structor: Refusing parent '%s': field '%s' range is invalid\n",
                            synth_struct.name.c_str(), field.name.c_str());
                        return BADADDR;
                    }
                    if (field.offset < *child_end &&
                        sub.parent_offset < *field_end) {
                        msg("Structor: Refusing parent '%s': child '%s' overlaps unmatched field '%s'\n",
                            synth_struct.name.c_str(), sub.structure.name.c_str(),
                            field.name.c_str());
                        return BADADDR;
                    }
                }
                SynthField nested;
                nested.offset = sub.parent_offset;
                nested.size = sub.structure.size;
                nested.type = sub_type;
                nested.semantic = SemanticType::NestedStruct;
                nested.confidence = TypeConfidence::Medium;
                nested.name = sub.field_name;
                synth_struct.size = std::max(
                    synth_struct.size,
                    static_cast<std::uint32_t>(*child_end));
                synth_struct.fields.push_back(std::move(nested));
            }
        }

        std::sort(synth_struct.fields.begin(), synth_struct.fields.end(),
                  [](const SynthField& a, const SynthField& b) {
                      if (a.offset != b.offset) return a.offset < b.offset;
                      if (a.is_bitfield != b.is_bitfield) return a.is_bitfield;
                      return a.bit_offset < b.bit_offset;
                  });
    }

    return create_struct(synth_struct);
}

tid_t StructurePersistence::create_vtable(SynthVTable& vtable) {
    std::optional<Transaction> owned_transaction;
    if (!transaction_active()) {
        owned_transaction = begin_transaction();
        if (!owned_transaction.has_value()) {
            return BADADDR;
        }
    }
    tid_t result = BADADDR;
    try {
        result = create_vtable_impl(vtable);
    } catch (...) {
        poison_transaction();
    }
    if (result == BADADDR) {
        poison_transaction();
    }
    if (result != BADADDR && transaction_poisoned()) {
        result = BADADDR;
        vtable.tid = BADADDR;
    }
    if (owned_transaction.has_value()) {
        if (result != BADADDR) {
            if (!owned_transaction->commit()) {
                result = BADADDR;
                vtable.tid = BADADDR;
            }
        } else if (!owned_transaction->rollback()) {
            msg("Structor: CRITICAL: implicit vtable transaction rollback failed\n");
        }
    }
    return result;
}

tid_t StructurePersistence::create_vtable_impl(SynthVTable& vtable) {
    if (vtable.name.empty() || vtable.slots.empty() ||
        vtable.slots.size() > MAX_VTABLE_SLOTS) {
        msg("Structor: Refusing empty vtable '%s'\n", vtable.name.c_str());
        return BADADDR;
    }

    qstring name = vtable.name;
    const tid_t prior_tid = get_named_type_tid(name.c_str());
    if (prior_tid != BADADDR &&
        (!is_generated_name(name, &vtable.naming) ||
         !is_structor_owned_type(prior_tid))) {
        name = make_unique_name(name.c_str());
        vtable.name = name;
    }

    // Create vtable type
    tinfo_t vtbl_type;
    udt_type_data_t udt;
    udt.is_union = false;
    udt.set_vftable(true);

    std::uint64_t total_size = 0;

    // Add slots
    for (const auto& slot : vtable.slots) {
        const auto slot_end = checked_interval_end(slot.offset, get_ptr_size());
        if (!is_valid_vtable_slot_offset(slot.offset) ||
            !slot_end.has_value() || *slot_end < 0 ||
            static_cast<std::uint64_t>(*slot_end) > MAX_STRUCT_SIZE) {
            msg("Structor: Refusing vtable '%s': invalid slot offset %lld\n",
                name.c_str(), static_cast<long long>(slot.offset));
            return BADADDR;
        }

        udm_t udm;
        udm.name = slot.name;
        udm.offset = static_cast<uint64>(slot.offset) * 8;  // Convert to bits

        if (!slot.func_type.empty()) {
            if (slot.func_type.is_func()) {
                if (!udm.type.create_ptr(slot.func_type)) {
                    return BADADDR;
                }
            } else {
                udm.type = slot.func_type;
            }
        } else {
            // Generic function pointer
            func_type_data_t ftd;
            ftd.rettype.create_simple_type(BTF_VOID);
            ftd.set_cc(CM_CC_UNKNOWN);
            tinfo_t func_type;
            if (!func_type.create_func(ftd) || !udm.type.create_ptr(func_type)) {
                return BADADDR;
            }
        }

        const size_t slot_type_size = udm.type.get_size();
        if (slot_type_size == BADSIZE || slot_type_size != get_ptr_size()) {
            msg("Structor: Refusing vtable '%s': slot '%s' is %zu bytes, expected %u\n",
                name.c_str(), slot.name.c_str(), slot_type_size, get_ptr_size());
            return BADADDR;
        }
        udm.size = get_ptr_size() * 8;
        total_size = std::max(
            total_size,
            static_cast<std::uint64_t>(slot.offset) + get_ptr_size());

        if (!slot.signature_hint.empty()) {
            udm.cmt = slot.signature_hint;
        }

        udt.push_back(udm);
    }

    const size_t requested_size = static_cast<size_t>(total_size);
    udt.total_size = requested_size;
    UdtLayoutExpectation expected =
        capture_layout_expectation(udt, 0, 0, requested_size);

    if (!vtbl_type.create_udt(udt)) {
        return BADADDR;
    }

    std::uint32_t vtable_alignment = 0;
    const size_t materialized_size = vtbl_type.get_size(&vtable_alignment);
    if (materialized_size == BADSIZE || materialized_size != requested_size ||
        vtable_alignment == 0) {
        msg("Structor: Refusing vtable '%s': materialized layout is "
            "size=%zu/%zu alignment=%u\n",
            name.c_str(), materialized_size, requested_size, vtable_alignment);
        return BADADDR;
    }
    expected.size = materialized_size;
    expected.effective_alignment = vtable_alignment;

    tinfo_code_t err = set_named_type_transactional(vtbl_type, name);
    if (err != TERR_OK) {
        return BADADDR;
    }

    tid_t tid = BADADDR;
    if (!verify_named_udt_layout(name, expected, &tid)) {
        return BADADDR;
    }
    qvector<ea_t> vtable_provenance;
    if (vtable.source_func != BADADDR) {
        vtable_provenance.push_back(vtable.source_func);
    }
    if (!store_provenance(tid, vtable_provenance)) {
        return BADADDR;
    }
    vtable.tid = tid;
    return tid;
}

bool StructurePersistence::update_struct(
    tid_t tid,
    const SynthStruct& synth_struct) {
    std::optional<Transaction> owned_transaction;
    if (!transaction_active()) {
        owned_transaction = begin_transaction();
        if (!owned_transaction.has_value()) {
            return false;
        }
    }
    bool result = false;
    try {
        result = update_struct_impl(tid, synth_struct);
    } catch (...) {
        poison_transaction();
    }
    if (!result) {
        poison_transaction();
    }
    if (result && transaction_poisoned()) {
        result = false;
    }
    if (owned_transaction.has_value()) {
        if (result) {
            result = owned_transaction->commit();
        } else if (!owned_transaction->rollback()) {
            msg("Structor: CRITICAL: implicit update transaction rollback failed\n");
        }
    }
    return result;
}

bool StructurePersistence::update_struct_impl(
    tid_t tid,
    const SynthStruct& synth_struct) {
    if (!validate_public_struct_shape(synth_struct)) {
        return false;
    }
    // Get the type by tid
    tinfo_t tif;
    if (!tif.get_type_by_tid(tid)) {
        return false;
    }

    // Get the name
    qstring name;
    tif.get_type_name(&name);
    if (name.empty()) {
        return false;
    }

    if (!generated_update_preserves_coverage(tif, name, synth_struct)) {
        return false;
    }

    std::uint32_t prior_alignment = 0;
    const size_t prior_size = tif.get_size(&prior_alignment);
    udt_type_data_t prior_udt;
    if (prior_size == BADSIZE || prior_alignment == 0 ||
        !tif.get_udt_details(&prior_udt)) {
        msg("Structor: Refusing update of '%s': prior layout cannot be snapshotted\n",
            name.c_str());
        return false;
    }
    // Recreate the structure with new fields
    udt_type_data_t udt;
    udt.is_union = false;
    udt.total_size = synth_struct.size;

    for (const auto& field : synth_struct.fields) {
        if (field.is_union_candidate && !field.union_members.empty()) {
            qvector<SynthField> members;
            members.reserve(field.union_members.size());
            for (const auto& alt : field.union_members) {
                SynthField member;
                member.name = alt.name;
                member.offset = alt.offset;
                member.size = alt.size;
                member.type = alt.type;
                member.comment = alt.comment;
                members.push_back(std::move(member));
            }

            qstring union_name = field.name.empty() ? qstring("union") : field.name;
            if (add_union_field(udt, field.offset, union_name, members) != BADADDR) {
                continue;
            }
        }

        udm_t udm;
        udm.name = field.name;

        if (field.is_bitfield) {
            udm.offset = static_cast<uint64>(field.offset) * 8 + field.bit_offset;
            udm.size = field.bit_size > 0 ? field.bit_size : field.size * 8;
            udm.type = make_bitfield_storage_type(field);
            if (udm.type.empty()) {
                msg("Structor: Refusing invalid bitfield '%s' (storage=%u width=%u)\n",
                    field.name.c_str(), field.size, field.bit_size);
                return false;
            }
        } else {
            udm.offset = static_cast<uint64>(field.offset) * 8;

            if (!field.type.empty()) {
                SynthField storage_field = field;
                storage_field.type = materialize_nested_type(name, field, field.type);
                udm.type = make_member_storage_type(storage_field);
                udm.size = field.size * 8;
            } else {
                tinfo_t byte_type;
                byte_type.create_simple_type(BT_INT8 | BTMT_CHAR);
                if (field.size > 1) {
                    if (!udm.type.create_array(byte_type, field.size)) {
                        msg("Structor: Failed to create storage for field '%s'\n",
                            field.name.c_str());
                        return false;
                    }
                } else {
                    udm.type = byte_type;
                }
                udm.size = field.size * 8;
            }
        }

        if (!field.comment.empty()) {
            udm.cmt = field.comment;
        }

        udt.push_back(udm);
    }

    if (!append_bitfield_tail_padding(udt, synth_struct.fields)) {
        msg("Structor: Refusing '%s': invalid bitfield storage layout\n",
            name.c_str());
        return false;
    }
    if (!configure_synth_udt_layout(udt, synth_struct, name)) {
        return false;
    }
    const UdtLayoutExpectation expected = capture_layout_expectation(
        udt, udt.pack, synth_struct.alignment, synth_struct.size);

    tinfo_t new_type;
    if (!new_type.create_udt(udt)) {
        return false;
    }

    tinfo_code_t err = set_named_type_transactional(new_type, name);
    if (err != TERR_OK) {
        msg("Structor: Transactional update of '%s' failed with save error %d\n",
            name.c_str(), err);
        return false;
    }

    tid_t verified_tid = BADADDR;
    if (!verify_named_udt_layout(name, expected, &verified_tid)) {
        msg("Structor: Transactional update of '%s' failed round-trip verification\n",
            name.c_str());
        return false;
    }

    qvector<ea_t> merged = get_provenance(tid);
    for (ea_t ea : synth_struct.provenance) {
        if (std::find(merged.begin(), merged.end(), ea) == merged.end()) {
            merged.push_back(ea);
        }
    }
    return set_provenance(verified_tid, merged);
}

bool StructurePersistence::delete_struct(tid_t tid) {
    try {
        if (transaction_active()) {
            msg("Structor: Refusing explicit deletion during a persistence transaction\n");
            return false;
        }
        tinfo_t tif;
        if (!tif.get_type_by_tid(tid)) {
            return false;
        }

        qstring name;
        tif.get_type_name(&name);
        if (name.empty()) {
            return false;
        }
        if (!is_structor_owned_type(tid)) {
            msg("Structor: Refusing to delete unowned type '%s'\n", name.c_str());
            return false;
        }

        tinfo_t snapshot = tif;
        if (!snapshot.detach()) {
            return false;
        }
        const qvector<ea_t> prior_provenance = load_provenance(tid);
        qstring node_name;
        node_name.sprnt("%s%llX", PROVENANCE_NETNODE_PREFIX,
                        static_cast<unsigned long long>(tid));
        netnode provenance_node(node_name.c_str(), 0, false);

        if (!del_named_type(nullptr, name.c_str(), NTF_TYPE)) {
            return false;
        }

        bool metadata_removed = false;
        try {
            if (provenance_node != BADNODE) {
                provenance_node.kill();
            }
            netnode observed(node_name.c_str(), 0, false);
            metadata_removed = observed == BADNODE;
        } catch (...) {
            metadata_removed = false;
        }
        if (metadata_removed) {
            return true;
        }

        bool rollback_succeeded = false;
        try {
            tinfo_t restored = snapshot;
            if (restored.set_named_type(
                    nullptr, name.c_str(), NTF_TYPE | NTF_REPLACE) == TERR_OK) {
                const tid_t restored_tid = get_named_type_tid(name.c_str());
                tinfo_t observed;
                rollback_succeeded = restored_tid != BADADDR &&
                    observed.get_type_by_tid(restored_tid) &&
                    observed.equals_to(snapshot) &&
                    store_provenance(restored_tid, prior_provenance);
            }
        } catch (...) {
            rollback_succeeded = false;
        }
        if (!rollback_succeeded) {
            msg("Structor: CRITICAL: failed to roll back deletion of '%s'\n",
                name.c_str());
        }
        return false;
    } catch (...) {
        return false;
    }
}

bool StructurePersistence::rename_struct(tid_t tid, const char* new_name) {
    if (transaction_active() || new_name == nullptr || new_name[0] == '\0') {
        return false;
    }
    tinfo_t tif;
    if (!tif.get_type_by_tid(tid)) {
        return false;
    }

    qstring old_name;
    if (!tif.get_type_name(&old_name) || old_name.empty() ||
        !is_structor_owned_type(tid)) {
        return false;
    }
    const qstring requested_name(new_name);
    if (old_name == requested_name) {
        return true;
    }
    const qvector<ea_t> prior_provenance = load_provenance(tid);

    const auto metadata_matches =
        [this, tid](const qstring& expected_name,
                    const qvector<ea_t>& expected_provenance) {
            tinfo_t current;
            qstring current_name;
            if (!current.get_type_by_tid(tid) ||
                !current.get_type_name(&current_name) ||
                current_name != expected_name ||
                !is_structor_owned_type(tid)) {
                return false;
            }
            const qvector<ea_t> current_provenance = load_provenance(tid);
            return current_provenance.size() == expected_provenance.size() &&
                std::equal(current_provenance.begin(),
                           current_provenance.end(),
                           expected_provenance.begin());
        };

    bool renamed = false;
    try {
        if (tif.rename_type(requested_name.c_str()) != TERR_OK) {
            return false;
        }
        renamed = true;
        if (store_provenance(tid, prior_provenance) &&
            metadata_matches(requested_name, prior_provenance)) {
            return true;
        }
    } catch (...) {
        // The rollback below covers exceptions after a successful host rename.
    }

    // A rename is successful only when its exact-name ownership marker and
    // provenance are durable. Restore both on any metadata failure/exception.
    bool rollback_succeeded = !renamed;
    try {
        tinfo_t renamed_type;
        if (renamed && renamed_type.get_type_by_tid(tid) &&
            renamed_type.rename_type(old_name.c_str()) == TERR_OK) {
            rollback_succeeded = store_provenance(tid, prior_provenance) &&
                metadata_matches(old_name, prior_provenance);
        }
    } catch (...) {
        rollback_succeeded = false;
    }
    if (!rollback_succeeded) {
        msg("Structor: CRITICAL: failed to roll back rename of '%s' to '%s'\n",
            requested_name.c_str(), old_name.c_str());
    }
    return false;
}

qvector<ea_t> StructurePersistence::get_provenance(tid_t tid) {
    return load_provenance(tid);
}

bool StructurePersistence::is_structor_owned_type(tid_t tid) const {
    if (tid == BADADDR) {
        return false;
    }
    qstring node_name;
    node_name.sprnt("%s%llX", PROVENANCE_NETNODE_PREFIX,
                    static_cast<unsigned long long>(tid));
    netnode node(node_name.c_str(), 0, false);
    if (node == BADNODE) {
        return false;
    }

    tinfo_t type;
    qstring current_name;
    if (!type.get_type_by_tid(tid) || !type.get_type_name(&current_name) ||
        current_name.empty()) {
        return false;
    }

    size_t marker_size = 0;
    void* marker = node.getblob(nullptr, &marker_size, 0, OWNERSHIP_TAG);
    if (marker == nullptr) {
        return false;
    }
    const size_t expected_size = current_name.length() + 1;
    const bool matches = marker_size == expected_size &&
        std::memcmp(marker, current_name.c_str(), expected_size) == 0;
    qfree(marker);
    return matches;
}

bool StructurePersistence::set_provenance(
    tid_t tid, const qvector<ea_t>& provenance) {
    if (!is_structor_owned_type(tid)) {
        return false;
    }
    return store_provenance(tid, provenance);
}

bool StructurePersistence::struct_exists(const char* name) {
    return get_named_type_tid(name) != BADADDR;
}

qstring StructurePersistence::make_unique_name(const char* base_name) {
    qstring name = base_name;

    if (!struct_exists(name.c_str())) {
        return name;
    }

    for (int i = 1; i < 10000; ++i) {
        qstring candidate;
        candidate.sprnt("%s_%d", base_name, i);
        if (!struct_exists(candidate.c_str())) {
            return candidate;
        }
    }

    // Fallback with timestamp
    qstring candidate;
    candidate.sprnt("%s_%llX", base_name, static_cast<unsigned long long>(time(nullptr)));
    return candidate;
}

bool StructurePersistence::store_provenance(
    tid_t tid, const qvector<ea_t>& provenance) {
    if (tid == BADADDR || provenance.size() > UINT32_MAX ||
        provenance.size() >
            (std::numeric_limits<std::size_t>::max() - sizeof(std::uint32_t)) /
                sizeof(ea_t)) {
        return false;
    }
    qstring node_name;
    node_name.sprnt("%s%llX", PROVENANCE_NETNODE_PREFIX, static_cast<unsigned long long>(tid));

    netnode node(node_name.c_str(), 0, true);
    if (node == BADNODE) return false;

    qvector<std::uint8_t> prior_ownership;
    qvector<std::uint8_t> prior_provenance;
    const bool had_prior_ownership =
        node.getblob(&prior_ownership, 0, OWNERSHIP_TAG) >= 0;
    const bool had_prior_provenance =
        node.getblob(&prior_provenance, 0, PROVENANCE_TAG) >= 0;

    const auto restore_metadata = [&]() noexcept {
        bool restored = true;
        try {
            if (had_prior_ownership) {
                restored = node.setblob(
                    prior_ownership.begin(), prior_ownership.size(),
                    0, OWNERSHIP_TAG) && restored;
            } else {
                (void)node.delblob(0, OWNERSHIP_TAG);
                qvector<std::uint8_t> absent;
                restored = node.getblob(&absent, 0, OWNERSHIP_TAG) < 0 && restored;
            }
            if (had_prior_provenance) {
                restored = node.setblob(
                    prior_provenance.begin(), prior_provenance.size(),
                    0, PROVENANCE_TAG) && restored;
            } else {
                (void)node.delblob(0, PROVENANCE_TAG);
                qvector<std::uint8_t> absent;
                restored = node.getblob(&absent, 0, PROVENANCE_TAG) < 0 && restored;
            }
        } catch (...) {
            restored = false;
        }
        if (!restored) {
            msg("Structor: CRITICAL: failed to restore ownership metadata for "
                "TID 0x%llX\n", static_cast<unsigned long long>(tid));
        }
        return restored;
    };

    try {
        // Bind automatic generated-type updates to both the TID and the exact
        // named type. A stale netnode left by an out-of-band deletion must not
        // claim a subsequently reused TID.
        tinfo_t owned_type;
        qstring owned_name;
        if (!owned_type.get_type_by_tid(tid) ||
            !owned_type.get_type_name(&owned_name) || owned_name.empty()) {
            return false;
        }

        // Serialize provenance
        qvector<char> blob;
        blob.reserve(provenance.size() * sizeof(ea_t) + 4);

        // Write count
        std::uint32_t count = provenance.size();
        const char* p = reinterpret_cast<const char*>(&count);
        for (size_t i = 0; i < sizeof(count); ++i) {
            blob.push_back(p[i]);
        }

        // Write EAs
        for (ea_t ea : provenance) {
            p = reinterpret_cast<const char*>(&ea);
            for (size_t i = 0; i < sizeof(ea); ++i) {
                blob.push_back(p[i]);
            }
        }

        if (!node.setblob(owned_name.c_str(), owned_name.length() + 1,
                          0, OWNERSHIP_TAG) ||
            !node.setblob(blob.begin(), blob.size(), 0, PROVENANCE_TAG)) {
            (void)restore_metadata();
            return false;
        }

        qvector<std::uint8_t> observed_ownership;
        qvector<std::uint8_t> observed_provenance;
        const bool verified =
            node.getblob(&observed_ownership, 0, OWNERSHIP_TAG) >= 0 &&
            observed_ownership.size() == owned_name.length() + 1 &&
            std::memcmp(observed_ownership.begin(), owned_name.c_str(),
                        observed_ownership.size()) == 0 &&
            node.getblob(&observed_provenance, 0, PROVENANCE_TAG) >= 0 &&
            observed_provenance.size() == blob.size() &&
            std::equal(observed_provenance.begin(), observed_provenance.end(),
                       reinterpret_cast<const std::uint8_t*>(blob.begin()));
        if (!verified) {
            (void)restore_metadata();
        }
        return verified;
    } catch (...) {
        (void)restore_metadata();
        return false;
    }
}

qvector<ea_t> StructurePersistence::load_provenance(tid_t tid) {
    qvector<ea_t> result;

    qstring node_name;
    node_name.sprnt("%s%llX", PROVENANCE_NETNODE_PREFIX, static_cast<unsigned long long>(tid));

    netnode node(node_name.c_str(), 0, false);
    if (node == BADNODE) return result;

    size_t blob_size = 0;
    void* blob = node.getblob(nullptr, &blob_size, 0, PROVENANCE_TAG);
    if (!blob || blob_size < 4) {
        if (blob) qfree(blob);
        return result;
    }

    const char* data = static_cast<const char*>(blob);

    // Read count
    std::uint32_t count;
    std::memcpy(&count, data, sizeof(count));
    data += sizeof(count);

    // Validate
    size_t expected_size = sizeof(count) + count * sizeof(ea_t);
    if (blob_size < expected_size) {
        qfree(blob);
        return result;
    }

    // Read EAs
    result.reserve(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        ea_t ea;
        std::memcpy(&ea, data, sizeof(ea));
        data += sizeof(ea);
        result.push_back(ea);
    }

    qfree(blob);
    return result;
}

tid_t StructurePersistence::create_union(
    const qstring& name,
    const qvector<SynthField>& members)
{
    std::optional<Transaction> owned_transaction;
    if (!transaction_active()) {
        owned_transaction = begin_transaction();
        if (!owned_transaction.has_value()) {
            return BADADDR;
        }
    }
    tid_t result = BADADDR;
    try {
        result = create_union_impl(name, members);
    } catch (...) {
        poison_transaction();
    }
    if (result == BADADDR) {
        poison_transaction();
    }
    if (result != BADADDR && transaction_poisoned()) {
        result = BADADDR;
    }
    if (owned_transaction.has_value()) {
        if (result != BADADDR) {
            if (!owned_transaction->commit()) {
                result = BADADDR;
            }
        } else if (!owned_transaction->rollback()) {
            msg("Structor: CRITICAL: implicit union transaction rollback failed\n");
        }
    }
    return result;
}

tid_t StructurePersistence::create_union_impl(
    const qstring& name,
    const qvector<SynthField>& members)
{
    if (name.empty() || members.empty() || members.size() > MAX_FIELDS) {
        return BADADDR;
    }

    // Generate unique name if needed
    const bool overlay_union = union_has_relative_members(members);

    qstring union_name = overlay_union ? make_internal_overlay_type_name(name) : name;
    const tid_t prior_tid = get_named_type_tid(union_name.c_str());
    if (prior_tid != BADADDR &&
        (!is_generated_name(union_name) ||
         !is_structor_owned_type(prior_tid))) {
        union_name = make_unique_union_name(union_name.c_str());
    }

    // Create the union type
    tinfo_t union_type;
    udt_type_data_t udt;
    udt.is_union = true;
    udt.total_size = compute_union_size(members);
    udt.pack = 0;
    const uint32_t union_size = udt.total_size;
    if (union_size == 0) {
        msg("Structor: Refusing to create union '%s' with invalid member bounds\n",
            union_name.c_str());
        return BADADDR;
    }

    // Add all members at offset 0 (union semantics)
    for (const auto& member : members) {
        udm_t udm;
        udm.name = member.name;
        udm.offset = 0;  // All union members start at offset 0

        if (overlay_union && member.offset != 0) {
            udm.type = create_overlay_array_type(member, union_size);
            if (udm.type.empty()) {
                udm.name = member.name;
                udm.type = create_overlay_view_type(union_name, member, union_size);
            }
            if (!udm.type.empty()) {
                const size_t view_size = udm.type.get_size();
                udm.size = view_size != BADSIZE ? view_size * 8 : union_size * 8;
            } else {
                udm.type = make_member_storage_type(member);
                const size_t type_size = udm.type.get_size();
                udm.size = type_size != BADSIZE ? type_size * 8 : member.size * 8;
            }
        } else if (!member.type.empty()) {
            udm.type = make_member_storage_type(member);
            udm.size = member.size * 8;
        } else {
            // Default to bytes array for unknown types
            tinfo_t byte_type;
            byte_type.create_simple_type(BT_INT8 | BTMT_CHAR);
            if (member.size > 1) {
                if (!udm.type.create_array(byte_type, member.size)) {
                    msg("Structor: Failed to create storage for union member '%s'\n",
                        member.name.c_str());
                    return BADADDR;
                }
            } else {
                udm.type = byte_type;
            }
            udm.size = member.size * 8;
        }

        if (!member.comment.empty()) {
            udm.cmt = member.comment;
        }

        udt.push_back(udm);
    }

    UdtLayoutExpectation expected =
        capture_layout_expectation(udt, 0, 0, union_size);

    // Create the union type
    if (!union_type.create_udt(udt, BTF_UNION)) {
        msg("Structor: Failed to create union type\n");
        return BADADDR;
    }

    std::uint32_t union_alignment = 0;
    const size_t materialized_size = union_type.get_size(&union_alignment);
    if (materialized_size == BADSIZE || materialized_size != union_size ||
        union_alignment == 0) {
        msg("Structor: Refusing union '%s': materialized layout is "
            "size=%zu/%u alignment=%u\n",
            union_name.c_str(), materialized_size, union_size, union_alignment);
        return BADADDR;
    }
    expected.size = materialized_size;
    expected.effective_alignment = union_alignment;

    // Save to local type library
    tinfo_code_t err = set_named_type_transactional(union_type, union_name);
    if (err != TERR_OK) {
        msg("Structor: Failed to save union type: %d\n", err);
        return BADADDR;
    }

    tid_t tid = BADADDR;
    if (!verify_named_udt_layout(union_name, expected, &tid)) {
        return BADADDR;
    }
    return store_provenance(tid, {}) ? tid : BADADDR;
}

tid_t StructurePersistence::add_union_field(
    udt_type_data_t& parent_udt,
    sval_t outer_offset,
    const qstring& union_name,
    const qvector<SynthField>& union_members)
{
    std::optional<Transaction> owned_transaction;
    if (!transaction_active()) {
        owned_transaction = begin_transaction();
        if (!owned_transaction.has_value()) {
            return BADADDR;
        }
    }
    tid_t result = BADADDR;
    try {
        result = add_union_field_impl(
            parent_udt, outer_offset, union_name, union_members);
    } catch (...) {
        poison_transaction();
    }
    if (result == BADADDR) {
        poison_transaction();
    }
    if (result != BADADDR && transaction_poisoned()) {
        result = BADADDR;
    }
    if (owned_transaction.has_value()) {
        if (result != BADADDR) {
            if (!owned_transaction->commit()) {
                result = BADADDR;
            }
        } else if (!owned_transaction->rollback()) {
            msg("Structor: CRITICAL: implicit union-field transaction rollback failed\n");
        }
    }
    return result;
}

tid_t StructurePersistence::add_union_field_impl(
    udt_type_data_t& parent_udt,
    sval_t outer_offset,
    const qstring& union_name,
    const qvector<SynthField>& union_members)
{
    if (union_members.empty() || union_members.size() > MAX_FIELDS ||
        outer_offset < 0 ||
        static_cast<std::uint64_t>(outer_offset) > MAX_STRUCT_SIZE) {
        return BADADDR;
    }
    const std::uint32_t union_size = compute_union_size(union_members);
    const auto union_end = checked_interval_end(outer_offset, union_size);
    if (union_size == 0 || !union_end.has_value() || *union_end < 0 ||
        static_cast<std::uint64_t>(*union_end) > MAX_STRUCT_SIZE) {
        return BADADDR;
    }

    // First, create the union type as a separate named type
    tid_t union_tid = create_union(union_name, union_members);
    if (union_tid == BADADDR) {
        return BADADDR;
    }

    // Get the union type info
    tinfo_t union_type;
    if (!union_type.get_type_by_tid(union_tid)) {
        return BADADDR;
    }

    // Add a field referencing the union type to the parent struct
    udm_t udm;
    udm.name = union_name;
    udm.offset = static_cast<uint64>(outer_offset) * 8;  // Convert to bits
    udm.type = union_type;
    udm.size = static_cast<std::uint64_t>(union_size) * 8;

    parent_udt.push_back(udm);

    return union_tid;
}

uint32_t StructurePersistence::compute_union_size(const qvector<SynthField>& members) {
    std::uint64_t max_end = 0;

    for (const auto& member : members) {
        if (member.offset < 0) {
            return 0;
        }

        uint32_t member_size = member.size;

        // If type is available, use its size instead
        if (!member.type.empty()) {
            size_t type_size = member.type.get_size();
            if (type_size != BADSIZE) {
                if (type_size == 0 || type_size > MAX_STRUCT_SIZE ||
                    type_size > std::numeric_limits<std::uint32_t>::max()) {
                    return 0;
                }
                member_size = static_cast<uint32_t>(type_size);
            }
        }

        if (member_size == 0) {
            return 0;
        }

        const std::uint64_t member_end =
            static_cast<std::uint64_t>(member.offset) + member_size;
        if (member_end > std::numeric_limits<uint32_t>::max() ||
            member_end > MAX_STRUCT_SIZE) {
            return 0;
        }
        max_end = std::max(max_end, member_end);
    }

    return static_cast<uint32_t>(max_end);
}

qstring StructurePersistence::make_unique_union_name(const char* base_name) {
    qstring name = base_name;

    if (!struct_exists(name.c_str())) {
        return name;
    }

    for (int i = 1; i < 10000; ++i) {
        qstring candidate;
        candidate.sprnt("%s_%d", base_name, i);
        if (!struct_exists(candidate.c_str())) {
            return candidate;
        }
    }

    // Fallback with timestamp
    qstring candidate;
    candidate.sprnt("%s_%llX", base_name, static_cast<unsigned long long>(time(nullptr)));
    return candidate;
}

tinfo_t StructurePersistence::create_raw_bytes_type(uint32_t size) {
    // Create uint8_t[size] type for irreconcilable regions
    tinfo_t element_type;
    element_type.create_simple_type(BT_INT8 | BTMT_USIGNED);

    tinfo_t array_type;
    array_type.create_array(element_type, size);

    return array_type;
}

tinfo_t StructurePersistence::create_bitfield_base_type(uint32_t size) {
    tinfo_t type;

    switch (size) {
        case 1:
            type.create_simple_type(BT_INT8 | BTMT_USIGNED);
            break;
        case 2:
            type.create_simple_type(BT_INT16 | BTMT_USIGNED);
            break;
        case 4:
            type.create_simple_type(BT_INT32 | BTMT_USIGNED);
            break;
        case 8:
            type.create_simple_type(BT_INT64 | BTMT_USIGNED);
            break;
        default:
            type.create_simple_type(BT_INT8 | BTMT_USIGNED);
            break;
    }

    return type;
}

tinfo_t StructurePersistence::create_bitmask_enum_type(
    const qstring& base_name,
    sval_t offset,
    uint32_t storage_size,
    const qvector<const SynthField*>& bitfields)
{
    if (bitfields.empty() || storage_size == 0 || storage_size > 8) {
        return tinfo_t();
    }

    qstring enum_name;
    enum_name.sprnt("%s_flags_%llX", base_name.c_str(),
                    static_cast<unsigned long long>(offset));
    if (struct_exists(enum_name.c_str())) {
        enum_name = make_unique_name(enum_name.c_str());
    }

    enum_type_data_t ei(BTE_ALWAYS | BTE_HEX);
    for (const auto* field : bitfields) {
        if (!field || field->bit_size == 0 || field->bit_size >= 64) {
            continue;
        }

        uint64 mask = ((uint64{1} << field->bit_size) - 1) << field->bit_offset;
        ei.push_back(edm_t(field->name.c_str(), mask));
    }

    if (ei.empty()) {
        return tinfo_t();
    }

    const PreparedNamedTypeWrite prepared = prepare_named_type_write(enum_name);
    if (!prepared.allowed) {
        return tinfo_t();
    }
    tid_t tid = BADADDR;
    try {
        tid = create_enum_type(
            enum_name.c_str(), ei, static_cast<int>(storage_size),
            type_unsigned, true, "Structor recovered bitmask flags");
    } catch (...) {
        try {
            mark_named_type_written(
                prepared, get_named_type_tid(enum_name.c_str()));
        } catch (...) {
            mark_named_type_written(prepared, BADADDR);
        }
        poison_transaction();
        return tinfo_t();
    }
    if (tid == BADADDR) {
        const tid_t partial_tid = get_named_type_tid(enum_name.c_str());
        if (partial_tid != BADADDR) {
            mark_named_type_written(prepared, partial_tid);
            poison_transaction();
        }
        return tinfo_t();
    }
    mark_named_type_written(prepared, tid);
    if (!store_provenance(tid, {})) {
        poison_transaction();
        return tinfo_t();
    }

    tinfo_t enum_type;
    if (!enum_type.get_type_by_tid(tid)) {
        poison_transaction();
        return tinfo_t();
    }
    return enum_type;
}

tinfo_t StructurePersistence::create_value_enum_type(
    const qstring& base_name,
    const SynthField& field)
{
    if (!field_supports_value_enum(field)) {
        return tinfo_t();
    }

    std::unordered_set<std::uint64_t> values;
    for (const auto& access : field.source_accesses) {
        for (auto value : access.observed_constants) {
            values.insert(value);
        }
    }

    if (!values.empty()) {
        msg("Structor:   Field '%s' has %zu observed constants\n",
            field.name.c_str(), values.size());
    }

    if (values.size() < 2 || values.size() > 32) {
        return tinfo_t();
    }

    const std::uint64_t max_value = max_storable_value(field.size);
    for (auto value : values) {
        if (value > max_value) {
            return tinfo_t();
        }
    }

    qstring enum_name;
    enum_name.sprnt("%s_%s_enum", base_name.c_str(), field.name.c_str());
    if (struct_exists(enum_name.c_str())) {
        enum_name = make_unique_name(enum_name.c_str());
    }

    enum_type_data_t ei(BTE_ALWAYS | BTE_HEX);
    qvector<std::uint64_t> ordered;
    ordered.reserve(values.size());
    for (auto value : values) {
        ordered.push_back(value);
    }
    std::sort(ordered.begin(), ordered.end());

    for (auto value : ordered) {
        qstring member_name;
        member_name.sprnt("value_%llX", static_cast<unsigned long long>(value));
        ei.push_back(edm_t(member_name.c_str(), value));
    }

    const PreparedNamedTypeWrite prepared = prepare_named_type_write(enum_name);
    if (!prepared.allowed) {
        return tinfo_t();
    }
    tid_t tid = BADADDR;
    try {
        tid = create_enum_type(
            enum_name.c_str(), ei, static_cast<int>(field.size),
            type_unsigned, true, "Structor recovered semantic constants");
    } catch (...) {
        try {
            mark_named_type_written(
                prepared, get_named_type_tid(enum_name.c_str()));
        } catch (...) {
            mark_named_type_written(prepared, BADADDR);
        }
        poison_transaction();
        return tinfo_t();
    }
    if (tid == BADADDR) {
        const tid_t partial_tid = get_named_type_tid(enum_name.c_str());
        if (partial_tid != BADADDR) {
            mark_named_type_written(prepared, partial_tid);
            poison_transaction();
        }
        msg("Structor:   Failed to create semantic enum '%s'\n", enum_name.c_str());
        return tinfo_t();
    }
    mark_named_type_written(prepared, tid);
    if (!store_provenance(tid, {})) {
        poison_transaction();
        return tinfo_t();
    }

    msg("Structor:   Created semantic enum '%s' for field '%s'\n",
        enum_name.c_str(), field.name.c_str());

    tinfo_t enum_type;
    if (!enum_type.get_type_by_tid(tid)) {
        poison_transaction();
        return tinfo_t();
    }
    return enum_type;
}

tinfo_t StructurePersistence::materialize_nested_type(
    const qstring& parent_name,
    const SynthField& field,
    const tinfo_t& type)
{
    if (type.empty()) {
        return type;
    }

    if (type.is_array()) {
        array_type_data_t atd;
        if (!type.get_array_details(&atd)) {
            return type;
        }

        if (atd.elem_type.is_struct() || atd.elem_type.is_union()) {
            qstring elem_name;
            atd.elem_type.get_type_name(&elem_name);
            if (elem_name.empty() || atd.elem_type.is_anonymous_udt()) {
                qstring nested_name = make_array_element_type_name(parent_name,
                                                                    field.name,
                                                                    field.offset);
                const tid_t prior_tid = get_named_type_tid(nested_name.c_str());
                if (prior_tid != BADADDR &&
                    (!is_generated_name(nested_name) ||
                     !is_structor_owned_type(prior_tid))) {
                    nested_name = make_unique_name(nested_name.c_str());
                }

                tinfo_t elem_type = atd.elem_type;
                std::uint32_t original_alignment = 0;
                const size_t original_size = elem_type.get_size(&original_alignment);
                udt_type_data_t elem_udt;
                if (elem_type.get_udt_details(&elem_udt)) {
                    apply_array_element_role_names(elem_udt);
                    elem_udt.pack = derive_anonymous_udt_pack(elem_udt);

                    const auto expected_members = capture_layout_expectation(
                        elem_udt, elem_udt.pack, original_alignment, original_size);
                    if (!elem_type.create_udt(elem_udt)) {
                        msg("Structor: Failed to materialize anonymous array element '%s'\n",
                            nested_name.c_str());
                        return type;
                    }

                    std::uint32_t rebuilt_alignment = 0;
                    const size_t rebuilt_size = elem_type.get_size(&rebuilt_alignment);
                    if (rebuilt_size == BADSIZE || rebuilt_alignment == 0 ||
                        (original_size != BADSIZE && rebuilt_size != original_size) ||
                        (original_alignment != 0 && rebuilt_alignment != original_alignment)) {
                        msg("Structor: Refusing anonymous array element '%s': "
                            "layout changed during materialization (size=%zu/%zu align=%u/%u)\n",
                            nested_name.c_str(), rebuilt_size, original_size,
                            rebuilt_alignment, original_alignment);
                        return type;
                    }

                    UdtLayoutExpectation expected = expected_members;
                    expected.size = rebuilt_size;
                    expected.effective_alignment = rebuilt_alignment;
                    if (set_named_type_transactional(elem_type, nested_name) != TERR_OK) {
                        msg("Structor: Failed to save anonymous array element '%s'\n",
                            nested_name.c_str());
                        return type;
                    }

                    tid_t tid = BADADDR;
                    if (!verify_named_udt_layout(nested_name, expected, &tid)) {
                        poison_transaction();
                        return type;
                    }
                    if (!store_provenance(tid, {})) {
                        poison_transaction();
                        return type;
                    }

                    tinfo_t named_elem;
                    if (!named_elem.get_type_by_tid(tid)) {
                        poison_transaction();
                        return type;
                    }

                    tinfo_t array_type;
                    if (!array_type.create_array(named_elem,
                                                 static_cast<uint32_t>(atd.nelems),
                                                 atd.base)) {
                        msg("Structor: Failed to rebuild array for element '%s'\n",
                            nested_name.c_str());
                        poison_transaction();
                        return type;
                    }
                    return array_type;
                }
            }
        }
    }

    return type;
}

SemanticType StructurePersistence::semantic_from_type(const tinfo_t& type) {
    return semantic_identity_from_type(type);
}

StructurePersistence::StructSignature StructurePersistence::build_signature(
    const SynthStruct& synth_struct)
{
    StructSignature sig;
    sig.size = synth_struct.size;
    sig.effective_alignment = synth_struct.alignment;
    const auto pack_code =
        persistence_invariants::ida_udt_pack_code(synth_struct.packing);
    if (pack_code.has_value() && sig.size != 0 && sig.effective_alignment != 0) {
        sig.pack_code = *pack_code;
        sig.layout_metadata_valid = true;
    }
    sig.fields.reserve(synth_struct.fields.size());

    for (const auto& field : synth_struct.fields) {
        if (field.is_padding) {
            continue;
        }

        FieldSignature fs;
        if (field.is_bitfield) {
            fs.offset = field.offset * 8 + field.bit_offset;
            fs.size = field.bit_size;
        } else {
            fs.offset = field.offset * 8;
            fs.size = field.size * 8;
        }
        if (field.is_union_candidate && !field.union_members.empty()) {
            fs.semantic = SemanticType::Unknown;
            fs.concrete_type = canonical_union_type_identity(field.union_members);
        } else {
            fs.semantic = field.semantic != SemanticType::Unknown
                ? field.semantic
                : semantic_from_type(field.type);
            fs.concrete_type = canonical_member_type_identity(
                field.type, fs.semantic, field.size);
        }

        if (fs.size == 0) {
            continue;
        }

        sig.fields.push_back(fs);
    }

    return sig;
}

bool StructurePersistence::build_signature_from_tinfo(
    const tinfo_t& tif,
    StructSignature& out)
{
    out = StructSignature{};

    if (!tif.is_struct()) {
        return false;
    }

    udt_type_data_t udt;
    if (!tif.get_udt_details(&udt)) {
        return false;
    }

    std::uint32_t effective_alignment = 0;
    const size_t type_size = tif.get_size(&effective_alignment);
    if (type_size == BADSIZE || type_size == 0 ||
        type_size > std::numeric_limits<std::uint32_t>::max() ||
        effective_alignment == 0 || udt.effalign != effective_alignment ||
        udt.pack > 5) {
        return false;
    }

    out.size = static_cast<std::uint32_t>(type_size);
    out.effective_alignment = effective_alignment;
    out.pack_code = udt.pack;
    out.layout_metadata_valid = true;

    out.fields.reserve(udt.size());

    for (const auto& member : udt) {
        const char* name = member.name.c_str();
        if (name && (strncmp(name, "__pad_", 6) == 0 || strncmp(name, "__raw_", 6) == 0)) {
            continue;
        }

        FieldSignature fs;
        fs.offset = static_cast<sval_t>(member.offset);
        fs.size = member.size;

        if (fs.size == 0 && !member.type.empty()) {
            size_t sz = member.type.get_size();
            if (sz != BADSIZE) {
                fs.size = static_cast<uint32_t>(sz * 8);
            }
        }

        if (member.type.is_union()) {
            fs.semantic = SemanticType::Unknown;
            fs.concrete_type = canonical_union_type_identity(member.type);
        } else {
            fs.semantic = semantic_from_type(member.type);
            fs.concrete_type = canonical_member_type_identity(
                member.type, fs.semantic,
                fs.size / 8 + (fs.size % 8 != 0 ? 1 : 0));
        }

        if (fs.size == 0) {
            continue;
        }

        out.fields.push_back(fs);
    }

    return !out.fields.empty();
}

double StructurePersistence::compute_similarity(
    const StructSignature& a,
    const StructSignature& b)
{
    if (!a.layout_metadata_valid || !b.layout_metadata_valid ||
        a.size != b.size ||
        a.effective_alignment != b.effective_alignment ||
        a.pack_code != b.pack_code ||
        a.fields.empty() || b.fields.empty()) {
        return 0.0;
    }

    struct Key {
        sval_t offset;
        uint32_t size;
        SemanticType semantic;
        std::string concrete_type;
    };

    struct KeyHash {
        size_t operator()(const Key& key) const noexcept {
            size_t h1 = std::hash<int64_t>{}(static_cast<int64_t>(key.offset));
            size_t h2 = std::hash<uint32_t>{}(key.size);
            size_t h3 = std::hash<int>{}(static_cast<int>(key.semantic));
            size_t h4 = std::hash<std::string>{}(key.concrete_type);
            return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3);
        }
    };

    struct KeyEq {
        bool operator()(const Key& lhs, const Key& rhs) const noexcept {
            return lhs.offset == rhs.offset &&
                   lhs.size == rhs.size &&
                   lhs.semantic == rhs.semantic &&
                   lhs.concrete_type == rhs.concrete_type;
        }
    };

    std::unordered_map<Key, int, KeyHash, KeyEq> counts;
    counts.reserve(a.fields.size());

    for (const auto& field : a.fields) {
        Key key{field.offset, field.size, field.semantic,
                field.concrete_type.c_str()};
        counts[key] += 1;
    }

    int matches = 0;
    for (const auto& field : b.fields) {
        Key key{field.offset, field.size, field.semantic,
                field.concrete_type.c_str()};
        auto it = counts.find(key);
        if (it != counts.end() && it->second > 0) {
            it->second -= 1;
            matches += 1;
        }
    }

    int total = static_cast<int>(a.fields.size() + b.fields.size() - matches);
    if (total <= 0) {
        return 0.0;
    }

    return static_cast<double>(matches) / static_cast<double>(total);
}

std::optional<std::tuple<tid_t, qstring, double>> StructurePersistence::find_reuse_candidate(
    const SynthStruct& synth_struct,
    double threshold)
{
    StructSignature synth_sig = build_signature(synth_struct);
    if (synth_sig.fields.empty()) {
        return std::nullopt;
    }

    til_t* til = get_idati();
    if (!til) {
        return std::nullopt;
    }

    uint32_t limit = get_ordinal_limit(til);
    struct ReuseSnapshot {
        tid_t tid = BADADDR;
        qstring name;
        StructSignature signature;
    };

    std::vector<ReuseSnapshot> snapshots;

    for (uint32_t ord = 1; ord < limit; ++ord) {
        tinfo_t tif;
        if (!tif.get_numbered_type(til, ord)) {
            continue;
        }

        if (!tif.is_struct()) {
            continue;
        }

        size_t type_size = tif.get_size();
        if (type_size == BADSIZE || type_size == 0) {
            continue;
        }

        if (synth_struct.size != 0 && type_size != synth_struct.size) {
            continue;
        }

        StructSignature sig;
        if (!build_signature_from_tinfo(tif, sig)) {
            continue;
        }

        if (!reuse_candidate_matches_function_members(synth_struct, tif)) {
            continue;
        }

        const char* type_name = get_numbered_type_name(til, ord);
        if (!type_name || type_name[0] == '\0') {
            continue;
        }

        tid_t tid = tif.get_tid();
        if (tid == BADADDR) {
            continue;
        }

        ReuseSnapshot snapshot;
        snapshot.tid = tid;
        snapshot.name = type_name;
        snapshot.signature = std::move(sig);
        snapshots.push_back(std::move(snapshot));
    }

    std::vector<double> scores(snapshots.size(), 0.0);
    algorithms::parallel_for_chunks(snapshots.size(), 64, [&](size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i) {
            scores[i] = compute_similarity(synth_sig, snapshots[i].signature);
        }
    });

    const auto best_index = persistence_invariants::unique_best_score_index(
        std::span<const double>(scores.data(), scores.size()), threshold);
    if (!best_index.has_value()) {
        return std::nullopt;
    }

    const auto& best = snapshots[*best_index];
    return std::make_optional(
        std::make_tuple(best.tid, best.name, scores[*best_index]));
}

} // namespace structor
