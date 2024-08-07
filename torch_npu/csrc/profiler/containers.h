#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <forward_list>
#include <new>
#include <utility>

namespace torch_npu {
namespace profiler {

constexpr size_t DEFAULT_BLOCK_SIZE = 1024;

template <
    typename T,
    size_t ChunkSize = DEFAULT_BLOCK_SIZE,
    template <typename U, size_t N> class block_t = std::array>
class AppendOnlyList {
public:
    using array_t = block_t<T, ChunkSize>;

    AppendOnlyList() : buffer_last_{buffer_.before_begin()} {}
    AppendOnlyList(const AppendOnlyList&) = delete;
    AppendOnlyList& operator=(const AppendOnlyList&) = delete;
    AppendOnlyList(AppendOnlyList&&) = default;
    AppendOnlyList& operator=(AppendOnlyList&&) = default;

    size_t size() const
    {
        return n_blocks_ * ChunkSize - (size_t)(end_ - next_);
    }

    template <class... Args>
    T* emplace_back(Args&&... args)
    {
        try_growup();
        if (std::is_trivially_destructible<T>::value && std::is_trivially_destructible<array_t>::value) {
            ::new ((void*)next_) T{std::forward<Args>(args)...};
        } else {
            *next_ = T{std::forward<Args>(args)...};
        }
        return next_++;
    }

    void clear()
    {
        buffer_.clear();
        buffer_last_ = buffer_.before_begin();
        n_blocks_ = 0;
        next_ = nullptr;
        end_ = nullptr;
    }

    struct Iterator {
        using iterator_category = std::forward_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = T;
        using pointer = T*;
        using reference = T&;

        Iterator() = default;
        Iterator(std::forward_list<array_t>& buffer, const size_t size)
            : block_{buffer.begin()}, size_{size} {}

        reference operator*() const
        {
            return *current_ptr();
        }

        pointer operator->()
        {
            return current_ptr();
        }

        Iterator& operator++()
        {
            if (!(++current_ % ChunkSize)) {
                block_++;
            }
            return *this;
        }

        Iterator operator++(int)
        {
            Iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        friend bool operator==(const Iterator& a, const Iterator& b)
        {
            return a.current_ptr() == b.current_ptr();
        }

        friend bool operator!=(const Iterator& a, const Iterator& b)
        {
            return a.current_ptr() != b.current_ptr();
        }

        std::pair<array_t*, size_t> address() const
        {
            if (current_ >= size_) {
                return {nullptr, 0};
            }
            return {&(*block_), current_ % ChunkSize};
        }

    private:
        T* current_ptr() const
        {
            auto ptr_pair = address();
            if (ptr_pair.first == nullptr) {
                return nullptr;
            }
            return ptr_pair.first->data() + ptr_pair.second;
        }

        typename std::forward_list<array_t>::iterator block_;
        size_t current_{0};
        size_t size_{0};
    };

    Iterator begin()
    {
        return Iterator(buffer_, size());
    }

    Iterator end()
    {
        return Iterator();
    }

protected:
    typename std::forward_list<array_t>::iterator buffer_last_;

private:
    void try_growup()
    {
        if (next_ == end_) {
            buffer_last_ = buffer_.emplace_after(buffer_last_);
            n_blocks_++;
            next_ = buffer_last_->data();
            end_ = next_ + ChunkSize;
        }
    }

    std::forward_list<array_t> buffer_;
    size_t n_blocks_{0};
    T* next_{nullptr};
    T* end_{nullptr};
};
} // profiler
} // torch_npu
