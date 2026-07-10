#pragma once

#include "synth_types.hpp"
#include "config.hpp"
#include "persistence_transaction.hpp"
#include "utils.hpp"
#include <netnode.hpp>
#include <memory>
#include <optional>
#include <tuple>
#include <vector>

namespace structor {

/// Handles persistence of synthesized structures to the IDB
class StructurePersistence {
private:
    struct TransactionOwnerToken {
        StructurePersistence* owner = nullptr;
    };

public:
    class Transaction {
    public:
        Transaction() noexcept = default;
        Transaction(const Transaction&) = delete;
        Transaction& operator=(const Transaction&) = delete;
        Transaction(Transaction&& other) noexcept;
        Transaction& operator=(Transaction&& other) noexcept;
        ~Transaction() noexcept;

        /// Commit the transaction. Returns false when a prior materialization
        /// failure poisoned the transaction; in that case commit performs a
        /// rollback instead of publishing a partially validated type graph.
        [[nodiscard]] bool commit() noexcept;
        [[nodiscard]] bool rollback() noexcept;
        [[nodiscard]] bool active() const noexcept;

    private:
        friend class StructurePersistence;
        explicit Transaction(
            const std::shared_ptr<TransactionOwnerToken>& owner) noexcept
            : owner_(owner), unresolved_(true) {}
        std::weak_ptr<TransactionOwnerToken> owner_;
        bool unresolved_ = false;
    };

    explicit StructurePersistence(const SynthOptions& opts = Config::instance().options())
        : options_(opts)
        , transaction_owner_token_(std::make_shared<TransactionOwnerToken>()) {
        transaction_owner_token_->owner = this;
    }
    ~StructurePersistence() noexcept;
    StructurePersistence(const StructurePersistence&) = delete;
    StructurePersistence& operator=(const StructurePersistence&) = delete;
    StructurePersistence(StructurePersistence&&) = delete;
    StructurePersistence& operator=(StructurePersistence&&) = delete;

    /// Begin an explicit root materialization transaction. The returned RAII
    /// handle rolls back unless commit() is called. A nested transaction is
    /// rejected with std::nullopt.
    [[nodiscard]] std::optional<Transaction> begin_transaction();
    [[nodiscard]] bool transaction_active() const noexcept;
    /// True when an active transaction has observed a post-write validation
    /// failure and therefore cannot be committed.
    [[nodiscard]] bool transaction_poisoned() const noexcept;

    /// Stage an auxiliary generated named type in the active transaction.
    /// An existing type is reused only when structurally identical; an
    /// incompatible or unowned name is never replaced.
    [[nodiscard]] bool stage_auxiliary_named_type(
        const qstring& name,
        const tinfo_t& definition,
        tinfo_t& out_named_type);

    /// Create a structure type in the IDB from synthesized structure
    [[nodiscard]] tid_t create_struct(SynthStruct& synth_struct);

    /// Create a structure with nested sub-structures
    [[nodiscard]] tid_t create_struct_with_substructs(
        SynthStruct& synth_struct,
        qvector<SubStructInfo>& sub_structs
    );

    /// Create a vtable structure in the IDB
    [[nodiscard]] tid_t create_vtable(SynthVTable& vtable);

    /// Create a union type in the IDB
    /// Returns the tid_t of the created union, or BADADDR on failure
    [[nodiscard]] tid_t create_union(
        const qstring& name,
        const qvector<SynthField>& members
    );

    /// Create a struct field that is itself an embedded union
    /// Union members are added at offset 0 within the union
    /// Returns the tid of the embedded union, or BADADDR on failure
    [[nodiscard]] tid_t add_union_field(
        udt_type_data_t& parent_udt,
        sval_t outer_offset,          // Offset of union within parent struct
        const qstring& union_name,
        const qvector<SynthField>& union_members
    );

    /// Compute union size (max of member sizes)
    [[nodiscard]] static uint32_t compute_union_size(const qvector<SynthField>& members);

    /// Update an existing structure with new fields
    [[nodiscard]] bool update_struct(tid_t tid, const SynthStruct& synth_struct);

    /// Delete a synthesized structure
    [[nodiscard]] bool delete_struct(tid_t tid);

    /// Rename a structure
    [[nodiscard]] bool rename_struct(tid_t tid, const char* new_name);

    /// Get provenance info for a structure
    [[nodiscard]] qvector<ea_t> get_provenance(tid_t tid);

    /// Set provenance info for a structure
    [[nodiscard]] bool set_provenance(
        tid_t tid, const qvector<ea_t>& provenance);

    /// Check if a structure name exists
    [[nodiscard]] bool struct_exists(const char* name);

    /// Generate a unique structure name
    [[nodiscard]] qstring make_unique_name(const char* base_name);

    /// Generate a unique union name
    [[nodiscard]] qstring make_unique_union_name(const char* base_name);

private:
    [[nodiscard]] tid_t create_struct_impl(SynthStruct& synth_struct);
    [[nodiscard]] tid_t create_struct_with_substructs_impl(
        SynthStruct& synth_struct,
        qvector<SubStructInfo>& sub_structs);
    [[nodiscard]] bool update_struct_impl(
        tid_t tid,
        const SynthStruct& synth_struct);
    [[nodiscard]] tid_t create_vtable_impl(SynthVTable& vtable);
    [[nodiscard]] tid_t create_union_impl(
        const qstring& name,
        const qvector<SynthField>& members);
    [[nodiscard]] tid_t add_union_field_impl(
        udt_type_data_t& parent_udt,
        sval_t outer_offset,
        const qstring& union_name,
        const qvector<SynthField>& union_members);

    bool add_struct_fields(tinfo_t& tif, const qvector<SynthField>& fields);
    bool add_vtable_slots(tinfo_t& tif, const qvector<VTableSlot>& slots);

    [[nodiscard]] bool store_provenance(
        tid_t tid, const qvector<ea_t>& provenance);
    qvector<ea_t> load_provenance(tid_t tid);
    [[nodiscard]] bool is_structor_owned_type(tid_t tid) const;

    struct FieldSignature {
        sval_t offset = 0;
        uint32_t size = 0;
        SemanticType semantic = SemanticType::Unknown;
        qstring concrete_type;
    };

    struct StructSignature {
        qvector<FieldSignature> fields;
        std::uint32_t size = 0;
        std::uint32_t effective_alignment = 0;
        std::uint8_t pack_code = 0;
        bool layout_metadata_valid = false;
    };

    [[nodiscard]] std::optional<std::tuple<tid_t, qstring, double>> find_reuse_candidate(
        const SynthStruct& synth_struct,
        double threshold
    );
    [[nodiscard]] static StructSignature build_signature(const SynthStruct& synth_struct);
    [[nodiscard]] static bool build_signature_from_tinfo(const tinfo_t& tif, StructSignature& out);
    [[nodiscard]] static double compute_similarity(const StructSignature& a, const StructSignature& b);
    [[nodiscard]] static SemanticType semantic_from_type(const tinfo_t& type);

    /// Create raw bytes field type for irreconcilable regions
    [[nodiscard]] static tinfo_t create_raw_bytes_type(uint32_t size);
    [[nodiscard]] static tinfo_t create_bitfield_base_type(uint32_t size);
    [[nodiscard]] tinfo_t create_bitmask_enum_type(
        const qstring& base_name,
        sval_t offset,
        uint32_t storage_size,
        const qvector<const SynthField*>& bitfields);
    [[nodiscard]] tinfo_t create_value_enum_type(
        const qstring& base_name,
        const SynthField& field);
    [[nodiscard]] tinfo_t materialize_nested_type(
        const qstring& parent_name,
        const SynthField& field,
        const tinfo_t& type);
    [[nodiscard]] tinfo_t create_overlay_view_type(
        const qstring& union_name,
        const SynthField& member,
        uint32_t union_size);

    struct NamedTypeSnapshot {
        bool valid = false;
        tid_t original_tid = BADADDR;
        tinfo_t type;
        qvector<ea_t> provenance;
    };

    struct TransactionState {
        persistence_invariants::PersistenceTransactionJournal journal;
        std::vector<NamedTypeSnapshot> snapshots;
        bool rolling_back = false;
        bool poisoned = false;
    };

    struct PreparedNamedTypeWrite {
        bool allowed = true;
        std::optional<std::size_t> journal_index;
    };

    [[nodiscard]] PreparedNamedTypeWrite prepare_named_type_write(
        const qstring& name);
    void mark_named_type_written(
        const PreparedNamedTypeWrite& prepared,
        tid_t current_tid);
    [[nodiscard]] tinfo_code_t set_named_type_transactional(
        tinfo_t& type,
        const qstring& name,
        int ntf_flags = NTF_TYPE | NTF_REPLACE);
    void poison_transaction() noexcept;
    [[nodiscard]] bool commit_transaction() noexcept;
    [[nodiscard]] bool rollback_transaction() noexcept;
    void kill_provenance_record(tid_t tid);

    SynthOptions options_;
    std::unique_ptr<TransactionState> transaction_state_;
    std::shared_ptr<TransactionOwnerToken> transaction_owner_token_;

    static constexpr const char* PROVENANCE_NETNODE_PREFIX = "$ structor_prov_";
    static constexpr nodeidx_t PROVENANCE_TAG = 'P';
    static constexpr nodeidx_t OWNERSHIP_TAG = 'O';
};

} // namespace structor
