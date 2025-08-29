# k_vs_time.py
import matplotlib.pyplot as plt

k_values = [3, 5, 7, 10]
avg_retrieval_time_s = [8.03, 19.51, 26.77, 38.17]  # per story

plt.figure()
plt.plot(k_values, avg_retrieval_time_s, marker='o')
plt.xlabel('Top-k')
plt.ylabel('Average retrieval time per story (s)')
plt.title('Effect of Top-k on retrieval time')
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

for x, y in zip(k_values, avg_retrieval_time_s):
    plt.text(x, y + 0.5, f"{y:.2f}", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('fig_topk_vs_time.png', dpi=200)
plt.show()
