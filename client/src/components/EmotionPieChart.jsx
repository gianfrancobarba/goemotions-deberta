import { PieChart, Pie, Cell, Tooltip, Legend } from 'recharts';

const COLORS = [
  '#8884d8', '#82ca9d', '#ffc658', '#ff8042',
  '#8dd1e1', '#a4de6c', '#d0ed57', '#ffbb28'
];

const EmotionPieChart = ({ emotions }) => {
  const data = Object.entries(emotions).map(([label, value]) => ({
    name: label,
    value: Math.round(value * 100) / 100,
  }));

  return (
    <PieChart width={400} height={300}>
      <Pie
        data={data}
        dataKey="value"
        nameKey="name"
        cx="50%"
        cy="50%"
        outerRadius={100}
        fill="#8884d8"
        label
      >
        {data.map((entry, index) => (
          <Cell key={entry.name} fill={COLORS[index % COLORS.length]} />
        ))}
      </Pie>
      <Tooltip />
      <Legend />
    </PieChart>
  );
};

export default EmotionPieChart;
