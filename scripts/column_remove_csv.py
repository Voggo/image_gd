
import csv
import sys

def main():
	if len(sys.argv) < 2:
		print("Usage: python column_remove_csv.py <csv_file>")
		sys.exit(1)
	csv_file = sys.argv[1]
	with open(csv_file, newline='') as f:
		reader = csv.reader(f)
		header = next(reader)
		print("Column headers:")
		for idx, col in enumerate(header):
			print(f"{idx}: {col}")
		indices = input("Enter indices of columns to delete (comma-separated): ")
		try:
			indices = [int(i.strip()) for i in indices.split(',') if i.strip().isdigit()]
		except Exception:
			print("Invalid input. Exiting.")
			sys.exit(1)
		def clean(cell):
			# Remove leading/trailing whitespace and extra quotes
			cell = cell.strip()
			cell = cell.replace('"', '')
			cell = cell.replace("'", '')
			cell = cell.strip()
			return cell
		new_header = [clean(col) for idx, col in enumerate(header) if idx not in indices]
		rows = []
		for row in reader:
			filtered = [clean(col) for idx, col in enumerate(row) if idx not in indices]
			rows.append(filtered)

		# Interpolate missing data
		import numpy as np
		def is_float(val):
			try:
				float(val)
				return True
			except:
				return False

		# Transpose rows to columns
		columns = list(zip(*rows))
		interpolated_columns = []
		for col in columns:
			# Interpolate numeric columns
			if all(is_float(cell) or cell == '' for cell in col):
				arr = np.array([float(cell) if cell != '' else np.nan for cell in col])
				# Linear interpolation
				nans = np.isnan(arr)
				arr[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), arr[~nans])
				interpolated_columns.append([str(val) for val in arr])
			else:
				# Fill forward for non-numeric columns
				filled = []
				last = ''
				for cell in col:
					if cell != '':
						last = cell
					filled.append(last)
				interpolated_columns.append(filled)
		# Transpose back to rows
		rows = list(zip(*interpolated_columns))
	output_file = csv_file.replace('.csv', '_removed.csv')
	with open(output_file, 'w', newline='') as f:
		writer = csv.writer(f)
		writer.writerow(new_header)
		writer.writerows(rows)
	print(f"Output written to {output_file}")

if __name__ == "__main__":
	main()
